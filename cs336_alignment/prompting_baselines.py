"""Evaluate prompting baselines (question_only / r1_zero / r1_zero_three_shot) on GSM8K.

See the assignment handout, section 3.4.
"""

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer

# The keys that `vllm_utils.generate_completions` actually forwards to the server.
# Anything else would be silently dropped, so we reject it up front.
SUPPORTED_SAMPLING_KEYS = {
    "temperature",
    "max_tokens",
    "n",
    "seed",
    "stop",
    "include_stop_str_in_output",
}


def validate_sampling_params(sampling_params: dict[str, Any]) -> None:
    """Fail loudly instead of letting unreachable parameters be ignored."""
    unsupported = set(sampling_params) - SUPPORTED_SAMPLING_KEYS
    if unsupported:
        raise ValueError(
            "vllm_utils.generate_completions only forwards "
            f"{sorted(SUPPORTED_SAMPLING_KEYS)}; these would be ignored: {sorted(unsupported)}"
        )
    if sampling_params.get("n", 1) != 1:
        raise ValueError(
            "The prompting baseline samples one response per question, so n must be 1."
        )


def load_json(data_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(data_path, mode="r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gsm8k(data_path: Path) -> list[dict[str, str]]:
    data = load_json(data_path)
    rows: list[dict[str, str]] = []
    for row in data:
        answer = row["answer"]
        if "####" in answer:
            ground_truth = answer.split("####")[-1].strip()
            rows.append(
                {
                    "question": row["question"],
                    "answer": answer,
                    "ground_truth": ground_truth,
                }
            )
    return rows


def build_prompts(questions: list[str], template: str) -> list[str]:
    return [template.format(question=question) for question in questions]


def generate_chunk_with_retry(
    server: VLLMServer,
    prompts: list[str],
    sampling_params: dict[str, Any],
    max_retries: int = 4,
) -> list[VLLMCompletion]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return server.generate_completions(
                prompts=prompts,
                sampling_params=sampling_params,
            )
        except Exception as error:  # URLError / timeouts / transient server errors
            last_error = error
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # 1s, 2s, 4s, 8s
    assert last_error is not None
    raise last_error


def generate_responses(
    server: VLLMServer,
    prompts: list[str],
    sampling_params: dict[str, Any],
    batch_size: int = 64,
    workers: int = 4,
    max_retries: int = 4,
) -> list[VLLMCompletion]:
    """Generate one completion per prompt, preserving prompt order."""
    validate_sampling_params(sampling_params)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")

    chunks = [
        prompts[start : start + batch_size]
        for start in range(0, len(prompts), batch_size)
    ]
    ordered: list[list[VLLMCompletion] | None] = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                generate_chunk_with_retry,
                server,
                chunk,
                sampling_params,
                max_retries,
            ): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            # Fill by chunk index: completion order is not chunk order.
            ordered[futures[future]] = future.result()

    completions = [completion for chunk in ordered if chunk for completion in chunk]
    if len(completions) != len(prompts):
        raise RuntimeError(
            f"Got {len(completions)} completions for {len(prompts)} prompts. "
            "Check that sampling_params['n'] == 1."
        )
    return completions


def score_zero_shot(responses: list[str], ground_truths: list[str]) -> list[dict[str, float]]:
    return [
        r1_zero_reward_fn(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths)
    ]


def score_question_only(responses: list[str], ground_truths: list[str]) -> list[dict[str, float]]:
    return [
        question_only_reward_fn(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths)
    ]


def score_responses(
    responses: list[str], ground_truths: list[str], prompt_type: str
) -> list[dict[str, float]]:
    if len(responses) != len(ground_truths):
        raise ValueError(
            f"Got {len(responses)} responses but {len(ground_truths)} ground truths."
        )
    if prompt_type in {"r1_zero", "r1_zero_three_shot"}:
        return score_zero_shot(responses, ground_truths)
    if prompt_type == "question_only":
        return score_question_only(responses, ground_truths)
    raise ValueError(f"Unknown prompt_type: {prompt_type}")


def summarize_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    """Count the four (format, answer) buckets.

    Keys are "11", "10", "01", "00" for (format_reward, answer_reward).
    """
    if not scores:
        raise ValueError("Cannot summarize an empty list of scores.")

    counts = {"11": 0, "10": 0, "01": 0, "00": 0}
    for score in scores:
        key = f"{int(score['format_reward'])}{int(score['answer_reward'])}"
        counts[key] += 1

    total = len(scores)
    assert sum(counts.values()) == total, "every score must land in exactly one bucket"

    accuracy = sum(score["answer_reward"] for score in scores) / total
    format_rate = sum(score["format_reward"] for score in scores) / total
    # 95% half-width for a proportion (binomial normal approximation).
    ci95 = 1.96 * math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / total)

    return {
        **counts,
        "total": total,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "ci95": ci95,
    }


def dump_records(
    path: Path,
    prompt_type: str,
    rows: list[dict[str, str]],
    completions: list[VLLMCompletion],
    scores: list[dict[str, float]],
) -> None:
    """Append one JSON line per example so the writeup does not need a re-run."""
    with open(path, mode="a", encoding="utf-8") as file:
        for row, completion, score in zip(rows, completions, scores):
            file.write(
                json.dumps(
                    {
                        "prompt_type": prompt_type,
                        "question": row["question"],
                        "ground_truth": row["ground_truth"],
                        "response": completion.text,
                        "finish_reason": completion.finish_reason,
                        "reward": score["reward"],
                        "format_reward": score["format_reward"],
                        "answer_reward": score["answer_reward"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def report_summary(
    prompt_type: str,
    num_examples: int,
    completions: list[VLLMCompletion],
    summary: dict[str, float],
) -> None:
    print(f"\n=== {prompt_type} ({num_examples} examples) ===")
    labels = {
        "11": "format correct, answer correct",
        "10": "format correct, answer incorrect",
        "01": "format incorrect, answer correct",
        "00": "format incorrect, answer incorrect",
    }
    for key, label in labels.items():
        count = int(summary[key])
        print(f"{label}: {count} ({100.0 * count / num_examples:.2f}%)")
    print(
        f"accuracy (mean answer_reward): {summary['accuracy']:.4f} "
        f"± {summary['ci95']:.4f} (95% CI)"
    )
    print(f"format rate: {summary['format_rate']:.4f}")
    truncated = sum(completion.finish_reason == "length" for completion in completions)
    print(f"truncated (finish_reason == 'length'): {truncated} / {len(completions)}")


def main():
    prompt_files = {
        "question_only": "question_only.prompt",
        "r1_zero": "r1_zero.prompt",
        "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
    }

    parser = argparse.ArgumentParser(
        description="Evaluate prompting baselines on GSM8K."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/gsm8k/test.jsonl"),
        help="Path to a GSM8K JSONL file.",
    )
    parser.add_argument(
        "--model-id",
        default="allenai/OLMo-2-0425-1B",
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for vLLM.")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Prompts per generation request."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent generation requests."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries per generation request before giving up.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optionally limit the number of evaluation examples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=Path("prompting_results.jsonl"),
        help="Where to write per-example records (overwritten at start of run).",
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=list(prompt_files),
        default=list(prompt_files),
        help="Prompt baselines to evaluate.",
    )
    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=10,
        help="Number of examples to print for manual inspection of each category.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.max_retries <= 0:
        parser.error("--max-retries must be positive.")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive when provided.")
    if args.examples_per_category < 0:
        parser.error("--examples-per-category cannot be negative.")

    rows = load_gsm8k(args.data_path)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    if not rows:
        raise ValueError(f"No GSM8K examples found in {args.data_path}.")

    prompts_dir = Path(__file__).resolve().parent / "prompts"
    ground_truths = [row["ground_truth"] for row in rows]
    base_sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "n": 1,
        "seed": args.seed,
    }

    # Start a clean record file for this run.
    args.dump_path.unlink(missing_ok=True)

    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.gpu,
        seed=args.seed,
        startup_timeout=600,
    )

    try:
        server.start()
        for prompt_type in args.prompt_types:
            template_path = prompts_dir / prompt_files[prompt_type]
            template = template_path.read_text(encoding="utf-8")
            prompts = build_prompts([row["question"] for row in rows], template)

            sampling_params = dict(base_sampling_params)
            if prompt_type in {"r1_zero", "r1_zero_three_shot"}:
                sampling_params["stop"] = ["</answer>"]
                sampling_params["include_stop_str_in_output"] = True

            completions = generate_responses(
                server=server,
                prompts=prompts,
                sampling_params=sampling_params,
                batch_size=args.batch_size,
                workers=args.workers,
                max_retries=args.max_retries,
            )
            responses = [completion.text for completion in completions]
            scores = score_responses(responses, ground_truths, prompt_type)
            summary = summarize_scores(scores)
            dump_records(args.dump_path, prompt_type, rows, completions, scores)

            report_summary(prompt_type, len(rows), completions, summary)

            category_specs = [
                (
                    "format correct, answer correct",
                    lambda score: score["format_reward"] == 1
                    and score["answer_reward"] == 1,
                    min(3, args.examples_per_category),
                ),
                (
                    "format correct, answer incorrect",
                    lambda score: score["format_reward"] == 1
                    and score["answer_reward"] == 0,
                    args.examples_per_category,
                ),
                (
                    "format incorrect, answer incorrect",
                    lambda score: score["format_reward"] == 0
                    and score["answer_reward"] == 0,
                    args.examples_per_category,
                ),
                (
                    "format incorrect, answer correct",
                    lambda score: score["format_reward"] == 0
                    and score["answer_reward"] == 1,
                    args.examples_per_category,
                ),
            ]
            for label, matches, limit in category_specs:
                example_indices = [
                    index for index, score in enumerate(scores) if matches(score)
                ][:limit]
                if not example_indices:
                    continue
                print(f"\nExamples: {label} (showing {len(example_indices)})")
                for index in example_indices:
                    print(f"Question: {rows[index]['question']}")
                    print(f"Prompt: {prompts[index]}")
                    print(f"Response: {responses[index]}")
                    print(f"Finish reason: {completions[index].finish_reason}")
                    print(f"Ground truth: {ground_truths[index]}")
                    print(f"Score: {scores[index]}\n")

        print(f"\nPer-example records written to {args.dump_path}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
