import regex as re
from typing import Any

def parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    marker = "answer is"
    if model_output is None:
        return None
    model_output = model_output.lower()
    if marker not in model_output:
        return None
    answer_str = model_output.split(marker)[-1].strip()
    match = re.search(r"\b([A-D])\b", answer_str, re.IGNORECASE)
    if match:
        real_ans = match.group(1).upper()
        # if real_ans in mmlu_example["options"]:
        #     return real_ans
        return real_ans
    return None

