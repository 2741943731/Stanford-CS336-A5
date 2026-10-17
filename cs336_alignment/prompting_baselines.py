import argparse
import json
from pathlib import Path
from typing import Any

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


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


def generate_responses(
    prompts: list[str],
    model_id: str,
    gpu: int,
    sampling_params: dict[str, Any],
    batch_size: int,
    server: VLLMServer | None = None,
) -> list[str]:
    owns_server = server is None
    if server is None:
        server = VLLMServer(model_id=model_id, gpu=gpu, startup_timeout=600)

    if owns_server:
        server.start()
    try:
        completions = server.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )
    finally:
        if owns_server:
            server.stop()

    if len(completions) != len(prompts):
        raise RuntimeError(
            f"Got {len(completions)} completions for {len(prompts)} prompts. "
            "Check that sampling_params['n'] == 1."
        )
    return [completion.text for completion in completions]


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


def summarize_scores(scores: list[dict[str, float]]) -> dict[str, int]:
    counts = {"cnt11": 0, "cnt10": 0, "cnt00": 0}
    for score in scores:
        if score["format_reward"] == 1 and score["answer_reward"] == 1:
            counts["cnt11"] += 1
        elif score["format_reward"] == 1 and score["answer_reward"] == 0:
            counts["cnt10"] += 1
        elif score["format_reward"] == 0 and score["answer_reward"] == 0:
            counts["cnt00"] += 1
    return counts


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
        "--batch-size", type=int, default=8, help="Prompts per generation request."
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

            responses = generate_responses(
                prompts=prompts,
                model_id=args.model_id,
                gpu=args.gpu,
                sampling_params=sampling_params,
                batch_size=args.batch_size,
                server=server,
            )
            scores = score_responses(responses, ground_truths, prompt_type)
            summary = summarize_scores(scores)

            print(f"\n=== {prompt_type} ({len(rows)} examples) ===")
            category_labels = {
                "cnt11": "format correct, answer correct",
                "cnt10": "format correct, answer incorrect",
                "cnt00": "format incorrect, answer reward 0",
            }
            for category, label in category_labels.items():
                count = summary[category]
                percentage = 100.0 * count / len(scores)
                print(f"{label}: {count} ({percentage:.2f}%)")

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
                    "format incorrect",
                    lambda score: score["format_reward"] == 0
                    and score["answer_reward"] == 0,
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
                    print(f"Ground truth: {ground_truths[index]}")
                    print(f"Score: {scores[index]}\n")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
