import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo_train_step_standard_on_policy import grpo_train_step
from cs336_alignment.prompting_baselines import load_gsm8k
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer

EVAL_SEED_OFFSET = 10_000

def format_prompts(rows: list[dict[str, str]], template: str) -> list[str]:
    return [template.format(question=row["question"]) for row in rows]


def prompt_batches(
    rows: list[dict[str, str]], batch_size: int, num_steps: int, seed: int
):
    """Yield `num_steps` batches of questions, reshuffling between epochs."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cursor = 0
    for _ in range(num_steps):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        yield [rows[index] for index in indices]

def sampling_params(
    temperature: float, max_tokens: int, n: int, seed: int, stop_at_answer: bool
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
    }
    if stop_at_answer:
        # Based on Dr. GRPO; see handout section 3.4.
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


def _call_with_retry(fn, max_retries: int):
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as error:  # URLError / timeouts / transient server errors
            last_error = error
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def sample_group(
    server: VLLMServer,
    prompt: str,
    group_size: int,
    params: dict[str, Any],
    max_retries: int,
) -> list[str]:
    """Draw `group_size` independent completions for a single prompt.

    One request per prompt with n=group_size (instead of one request holding
    group_size copies of the same prompt) so that vLLM seeds the samples
    independently and the returned order is simply index 0..n-1.
    """
    params = dict(params, n=group_size)

    def call() -> list[VLLMCompletion]:
        return server.generate_completions(prompts=[prompt], sampling_params=params)

    completions = _call_with_retry(call, max_retries)
    if len(completions) != group_size:
        raise RuntimeError(
            f"Expected {group_size} completions for one prompt, got {len(completions)}."
        )
    return [completion.text for completion in completions]


def generate_rollouts(
    server: VLLMServer,
    prompts: list[str],
    group_size: int,
    params: dict[str, Any],
    workers: int,
    max_retries: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Returns (repeated_prompts, rollout_responses), both prompt-major."""
    per_prompt_params = dict(params, seed=seed)
    groups: list[list[str] | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(sample_group, server, prompt, group_size, per_prompt_params, max_retries): index
            for index, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            groups[futures[future]] = future.result()

    repeated_prompts = [prompt for prompt in prompts for _ in range(group_size)]
    rollout_responses = [response for group in groups if group for response in group]
    if len(rollout_responses) != len(prompts) * group_size:
        raise RuntimeError("Rollout generation lost or duplicated samples.")
    return repeated_prompts, rollout_responses


def generate_one_per_prompt(
    server: VLLMServer,
    prompts: list[str],
    params: dict[str, Any],
    batch_size: int,
    workers: int,
    max_retries: int,
) -> list[VLLMCompletion]:
    """Evaluation: one completion per prompt, in prompt order."""
    chunks = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    ordered: list[list[VLLMCompletion] | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _call_with_retry,
                lambda chunk=chunk: server.generate_completions(
                    prompts=chunk, sampling_params=params
                ),
                max_retries,
            ): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()

    completions = [completion for chunk in ordered if chunk for completion in chunk]
    if len(completions) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} completions, got {len(completions)}."
        )
    return completions


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    server: VLLMServer,
    rows: list[dict[str, str]],
    template: str,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, float]:
    prompts = format_prompts(rows, template)
    params = sampling_params(
        args.sampling_temperature,
        args.sampling_max_tokens,
        1,
        args.seed + EVAL_SEED_OFFSET,
        args.stop_at_answer,
    )
    completions = generate_one_per_prompt(
        server, prompts, params, args.eval_batch_size, args.workers, args.max_retries
    )
    responses = [completion.text for completion in completions]
    scores = [
        r1_zero_reward_fn(response, row["ground_truth"])
        for response, row in zip(responses, rows)
    ]
    lengths = [
        len(tokenizer.encode(response, add_special_tokens=False))
        for response in responses
    ]
    num = len(rows)
    truncated = sum(completion.finish_reason == "length" for completion in completions)
    return {
        "val/reward": sum(score["reward"] for score in scores) / num,
        "val/format_reward": sum(score["format_reward"] for score in scores) / num,
        "val/answer_reward": sum(score["answer_reward"] for score in scores) / num,
        "val/response_length": sum(lengths) / num,
        "val/truncated_fraction": truncated / num,
    }


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def maybe_init_wandb(args: argparse.Namespace):
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[warn] wandb is not installed; running without it.")
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )


def log_metrics(
    run,
    metrics: dict[str, float],
    step: int,
    metrics_path: Path,
) -> None:
    with open(metrics_path, "a", encoding="utf-8") as file:
        file.write(json.dumps({"step": step, **metrics}) + "\n")
    if run is not None:
        run.log(metrics, step=step)
    printable = "  ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"[step {step:>3}] {printable}", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training loop on GSM8K.")

    # Data / model
    parser.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-path", default="data/gsm8k/train.jsonl")
    parser.add_argument("--val-path", default="data/gsm8k/test.jsonl")
    parser.add_argument(
        "--prompt-path",
        default="cs336_alignment/prompts/r1_zero.prompt",
        help="Prompt template with a {question} placeholder.",
    )
    parser.add_argument("--output-dir", default="runs/grpo-r1zero")
    parser.add_argument("--policy-device", default="cuda:7")
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    # Data sizes
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=1024)

    # Training hyperparameters (handout defaults)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Sampling
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument(
        "--stop-at-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop generation at </answer> (r1_zero style prompts).",
    )

    # Algorithm variants (section 5 / 6)
    parser.add_argument(
        "--baseline", choices=["mean", "none"], default="mean"
    )
    parser.add_argument(
        "--advantage-normalizer", choices=["std", "none", "mean"], default="std"
    )
    parser.add_argument(
        "--loss-normalization", choices=["sequence", "constant"], default="sequence"
    )
    parser.add_argument(
        "--normalization-constant",
        type=int,
        default=None,
        help="Defaults to rollout_batch_size * sampling_max_tokens when needed.",
    )
    parser.add_argument(
        "--importance-reweighting-method",
        choices=["none", "noclip", "grpo", "gspo"],
        default="none",
        help="On-policy training only supports 'none' for now.",
    )
    parser.add_argument("--cliprange", type=float, default=None)

    # Evaluation / logging
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-examples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--log-rollouts-every", type=int, default=40)
    parser.add_argument("--rollouts-to-log", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cs336-a5")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--save-every", type=int, default=0, help="0 disables.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size must be a multiple of group_size.")
    if args.rollout_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by gradient_accumulation_steps."
        )
    if args.train_batch_size != args.rollout_batch_size:
        raise ValueError(
            "This script updates once per rollout batch, so train_batch_size "
            "must equal rollout_batch_size."
        )
    if args.importance_reweighting_method != "none":
        raise NotImplementedError(
            "This script implements on-policy GRPO only; off-policy variants "
            "(section 6) additionally need old_log_probs from the sampling policy."
        )
    if args.loss_normalization == "constant":
        if args.normalization_constant is None:
            # Dr. GRPO uses Z = B * G * L (handout section 5.1).
            args.normalization_constant = (
                args.rollout_batch_size * args.sampling_max_tokens
            )
        if args.normalization_constant <= 0:
            raise ValueError("normalization_constant must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    rollouts_path = output_dir / "rollouts.jsonl"
    rollouts_path.unlink(missing_ok=True)

    train_rows = load_gsm8k(Path(args.train_path))[: args.n_train_examples]
    val_rows = load_gsm8k(Path(args.val_path))[: args.n_val_examples]
    eval_rows = val_rows[: args.eval_examples]
    if not train_rows or not eval_rows:
        raise ValueError("Empty train or validation set.")
    template = Path(args.prompt_path).read_text(encoding="utf-8")

    prompts_per_step = args.rollout_batch_size // args.group_size
    print(
        f"train rows={len(train_rows)} val rows={len(eval_rows)} "
        f"prompts/step={prompts_per_step} samples/step={args.rollout_batch_size}"
    )

    torch.cuda.set_device(torch.device(args.policy_device))
    policy, tokenizer = get_model_and_tokenizer(args.model_id, args.policy_device)
    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    run = maybe_init_wandb(args)
    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.vllm_gpu,
        seed=args.seed,
    )

    try:
        server.start()
        server.init_weight_sync(args.policy_device)

        batches = prompt_batches(
            train_rows, prompts_per_step, args.num_rollout_steps, args.seed
        )
        for step in range(args.num_rollout_steps):
            batch = next(batches)
            prompt_strs = format_prompts(batch, template)
            ground_truths = [row["ground_truth"] for row in batch]

            # 1. Make vLLM use exactly the weights we are about to train.
            server.sync_policy_weights(policy)

            # 2. G independent rollouts per question.
            params = sampling_params(
                args.sampling_temperature,
                args.sampling_max_tokens,
                args.group_size,
                args.seed + step,
                args.stop_at_answer,
            )
            repeated_prompts, responses = generate_rollouts(
                server,
                prompt_strs,
                args.group_size,
                params,
                args.workers,
                args.max_retries,
                args.seed + step,
            )
            repeated_ground_truths = [
                gt for gt in ground_truths for _ in range(args.group_size)
            ]

            # 3. Evaluate with the current (pre-update) vLLM weights.
            metrics: dict[str, float] = {}
            if step % args.eval_every == 0:
                metrics.update(
                    evaluate(
                        server,
                        eval_rows,
                        template,
                        tokenizer,
                        args,
                    )
                )

            # 4. Rewards -> advantages -> gradient accumulation -> optimizer step.
            loss, metadata = grpo_train_step(
                policy,
                tokenizer,
                optimizer,
                args.gradient_accumulation_steps,
                args.max_grad_norm,
                r1_zero_reward_fn,
                repeated_prompts,
                responses,
                repeated_ground_truths,
                args.group_size,
                baseline=args.baseline,
                advantage_normalizer=args.advantage_normalizer,
                importance_reweighting_method=args.importance_reweighting_method,
                cliprange=args.cliprange,
                loss_normalization=args.loss_normalization,
                normalization_constant=args.normalization_constant,
            )

            response_lengths = [
                len(tokenizer.encode(response, add_special_tokens=False))
                for response in responses
            ]
            metrics.update(
                {
                    "train/loss": metadata["mean_loss"],
                    "train/batch_loss": float(loss.detach()),
                    "train/grad_norm": metadata["grad_norm"],
                    "train/token_entropy": metadata["mean_entropy"],
                    "train/reward": metadata["mean_reward"],
                    "train/reward_std": metadata.get("std"),
                    "train/format_reward": metadata["mean_format_reward"],
                    "train/response_length": sum(response_lengths) / len(response_lengths),
                }
            )
            # Drop missing/NaN values so metrics.jsonl stays valid JSON.
            metrics = {
                key: value
                for key, value in metrics.items()
                if value is not None
                and not (isinstance(value, float) and math.isnan(value))
            }
            log_metrics(run, metrics, step, metrics_path)

            # 5. Keep a few rollouts around for eyeballing.
            if step % args.log_rollouts_every == 0:
                keep = min(args.rollouts_to_log, len(responses))
                with open(rollouts_path, "a", encoding="utf-8") as file:
                    for index in range(keep):
                        file.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "question": batch[index // args.group_size]["question"],
                                    "response": responses[index],
                                    "ground_truth": repeated_ground_truths[index],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                for index in range(min(2, keep)):
                    print(
                        f"--- rollout @ step {step} ---\n"
                        f"Q: {batch[index // args.group_size]['question']}\n"
                        f"A: {responses[index][-600:]}\n"
                    )

            if args.save_every and (step + 1) % args.save_every == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step + 1}"
                policy.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"saved checkpoint to {checkpoint_dir}")

        final_dir = output_dir / "checkpoint-last"
        policy.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"saved final checkpoint to {final_dir}")
    finally:
        server.stop()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
