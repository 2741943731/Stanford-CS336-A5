import torch
from typing import Callable, Literal

def compute_group_normalized_rewards_grpo(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    
    if baseline not in ["mean", "none"]:
        raise ValueError("baseline must be either 'mean' or 'none'")
    if advantage_normalizer not in ["std", "none", "mean"]:
        raise ValueError("advantage_normalizer must be either 'std', 'none', or 'mean'")

    
    raw_rewards = raw_rewards.view(-1, group_size)
    mean = raw_rewards.mean(dim=1, keepdim=True)
    std = raw_rewards.std(dim=1, keepdim=True)
    if baseline == "mean":
        normalized_rewards = raw_rewards - mean
    else:
        normalized_rewards = raw_rewards

    if advantage_normalizer == "std":
        normalized_rewards = normalized_rewards / (std + advantage_eps)
    elif advantage_normalizer == "mean":
        normalized_rewards = normalized_rewards / (mean + advantage_eps)

    return normalized_rewards.view(-1), {
        "mean": raw_rewards.mean().item(),
        "std": raw_rewards.std().item(),
        "max": raw_rewards.max().item(),
        "min": raw_rewards.min().item(),
    }