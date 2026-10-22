import torch
from torch import Tensor
from typing import Callable, Literal
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from cs336_alignment.scoring import get_response_log_probs
from cs336_alignment.compute_rollot_rewards import compute_rollout_rewards
from cs336_alignment.data_processing import tokenize_prompts
from cs336_alignment.compute_policy_gradient_loss import compute_policy_gradient_loss
from cs336_alignment.aggregate_loss_across_microbatch_sequence import aggregate_loss_across_microbatch
from cs336_alignment.compute_group_normalized_rewards_grpo import compute_group_normalized_rewards_grpo

def grpo_train_step(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: Optimizer,
        gradient_accumulation_steps: int,
        max_grad_norm: float | None,
        reward_fn: Callable[[str, str], dict[str, float]],
        repeated_promots: list[str],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
        group_size: int,
        baseline: Literal["mean", "none"] = "mean",
        advantage_eps: float = 1e-6,
        advantage_normalizer: Literal["std", "none", "mean"] = "std",
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: Tensor | None = None,
        cliprange: float | None = None,
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None,
) -> tuple[Tensor, dict[str, Tensor | float]]:
    batch_loss = torch.tensor(0.0, device=model.device)
    raw_rewards, rollout_metadata = compute_rollout_rewards(
        reward_fn, 
        rollout_responses, 
        repeated_ground_truths)
    normalized_rewards, normalized_metadata = compute_group_normalized_rewards_grpo(
        raw_rewards, 
        group_size, 
        baseline, 
        advantage_eps, 
        advantage_normalizer)
    batch_size = len(repeated_promots)
    minibatch_size = batch_size // gradient_accumulation_steps
    input_ids, labels, response_mask = tokenize_prompts(
        repeated_promots,
        rollout_responses,
        tokenizer).values()
    losses = []
    entropy_sum = 0.0
    entropy_tokens = 0.0
    for i in range(0, batch_size, minibatch_size):
        minibatch_input_ids = input_ids[i: i + minibatch_size, :]
        minibatch_labels = labels[i: i + minibatch_size, :]
        minibatch_response_mask = response_mask[i: i + minibatch_size, :]

        minibatch_normalized_rewards = normalized_rewards[i: i + minibatch_size]
        minibatch_old_log_probs = old_log_probs[i: i + minibatch_size, :] if old_log_probs is not None else None
        log_probs, token_entropy = get_response_log_probs(
            model, 
            minibatch_input_ids, 
            minibatch_labels, 
            return_token_entropy=True,
            need_grad=True
            ).values()
        per_token_policy_gradient_loss, _ = compute_policy_gradient_loss(
            minibatch_normalized_rewards,
            log_probs, 
            importance_reweighting_method, 
            minibatch_old_log_probs, 
            cliprange, 
            minibatch_response_mask)
        loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_policy_gradient_loss,
            mask=minibatch_response_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant
        )
        losses.append(loss.item())
        if token_entropy is not None:
            entropy_sum += float((token_entropy.detach() * minibatch_response_mask).sum())
            entropy_tokens += float(minibatch_response_mask.sum())
        if not normalization_constant:
            loss = loss / gradient_accumulation_steps
        batch_loss += loss
        loss.backward()
    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss, {"loss": batch_loss.item(), 
                        "mean_reward": rollout_metadata["mean_reward"], 
                        "mean_format_reward": rollout_metadata["mean_format_reward"], 
                        **normalized_metadata, 
                        "mean_loss": sum(losses) / len(losses),
                        "grad_norm": float(grad_norm) if grad_norm is not None else float("nan"),
                        "mean_entropy": entropy_sum / entropy_tokens if entropy_tokens else float("nan")}