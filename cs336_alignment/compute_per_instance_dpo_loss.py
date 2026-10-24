import torch
from pathlib import Path
from transformers import PreTrainedTokenizerBase

def response_token_log_probs(
    model: torch.nn.Module, 
    tokenizer: PreTrainedTokenizerBase, 
    prompt: str, 
    response: str
) -> torch.Tensor:
    template = Path(
        "cs336_alignment/prompts_safety/alpaca_sft.prompt"
    ).read_text(encoding="utf-8").rstrip("\n")

    full_text = template.format(instruction=prompt, response=response)
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    ids.append(tokenizer.eos_token_id)
    full_ids = torch.tensor([ids], device=next(model.parameters()).device)

    logits = model(full_ids).logits  # 1 s v
    log_logits = torch.nn.functional.log_softmax(logits, dim=-1)  # 1 s v
    next_token_ids = full_ids[:, 1:]  # 1 s-1
    log_probs = torch.gather(log_logits[:, :-1, :], dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(-1)  # 1 s-1
    # prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # 虽然要求计算的事response的log_probs但是相同模型的prompt部分的probs也一样，所以不必截取，相减会抵消
    return log_probs

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    chosen_log_probs = response_token_log_probs(lm, tokenizer, prompt, response_chosen)
    rejected_log_probs = response_token_log_probs(lm, tokenizer, prompt, response_rejected)
    chosen_ref_log_probs = response_token_log_probs(lm_ref, tokenizer, prompt, response_chosen)
    rejected_ref_log_probs = response_token_log_probs(lm_ref, tokenizer, prompt, response_rejected)
    chosen_log_ratio = chosen_log_probs.sum() - chosen_ref_log_probs.sum()
    rejected_log_ratio = rejected_log_probs.sum() - rejected_ref_log_probs.sum()
    loss = -torch.log(torch.sigmoid(beta * (chosen_log_ratio - rejected_log_ratio)))
    return loss

