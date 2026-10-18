import torch
from torch import Tensor
from typing import Callable

def compute_rollout_rewards(
        reward_fn: Callable[[str, str], dict[str, float]],
        rollout_responses: list[str],
        repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards = torch.zeros(len(rollout_responses), dtype=torch.float32)
    format_rewards = torch.zeros(len(rollout_responses), dtype=torch.float32)
    for i, (response, ground_truth) in enumerate(zip(rollout_responses, repeated_ground_truths)):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards[i] = reward_dict["reward"]
        format_rewards[i] = reward_dict["format_reward"]
    metadata = {
        "mean_reward": raw_rewards.mean().item(), 
        "mean_format_reward": format_rewards.mean().item()
    }
    return raw_rewards, metadata