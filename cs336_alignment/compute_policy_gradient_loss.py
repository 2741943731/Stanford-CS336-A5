import torch
from torch import Tensor
from typing import Callable, Literal

def compute_policy_gradient_loss(
        raw_rewards_or_advantages: Tensor,
        policy_log_probs: Tensor,
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
        old_log_probs: Tensor | None = None,
        cliprange: float | None = None,
        response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    raw_rewards_or_advantages = raw_rewards_or_advantages.view(-1, 1)  # b -> b 1
    # per_token_loss = -raw_rewards_or_advantages * policy_log_probs
    if importance_reweighting_method == "none":
        per_token_loss = -raw_rewards_or_advantages * policy_log_probs
        return per_token_loss, {}
    elif importance_reweighting_method == "noclip" or importance_reweighting_method == "grpo":
        r_b_t = torch.exp(policy_log_probs - old_log_probs)
        if importance_reweighting_method == "grpo":
            clipped_r_b_t = torch.clamp(r_b_t, 1-cliprange, 1+cliprange)
            per_token_loss = -torch.min(raw_rewards_or_advantages * r_b_t, raw_rewards_or_advantages * clipped_r_b_t)
            return per_token_loss, {}
        else:
            per_token_loss = -raw_rewards_or_advantages * r_b_t
            return per_token_loss, {}
    elif importance_reweighting_method == "gspo":
        s_b = torch.exp(
            (response_mask * policy_log_probs - response_mask * old_log_probs).sum(dim=-1) / response_mask.sum(dim=-1) # b
        )
        clipped_s_b = torch.clamp(s_b, 1-cliprange, 1+cliprange) # b
        # per_token_loss = -raw_rewards_or_advantages.squeeze(-1) * torch.min(s_b.unsqueeze(-1), clipped_s_b.unsqueeze(-1))
        per_token_loss = -torch.min(raw_rewards_or_advantages.squeeze(-1) * s_b, raw_rewards_or_advantages.squeeze(-1) * clipped_s_b)
        per_token_loss = per_token_loss.unsqueeze(-1).repeat(1, policy_log_probs.shape[-1])
        return per_token_loss, {}
    else:
        raise ValueError("check wtf u input")
