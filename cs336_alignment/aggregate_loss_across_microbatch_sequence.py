import torch
from torch import Tensor
from typing import Callable, Literal

def aggregate_loss_across_microbatch(
        per_token_policy_gradient_loss: Tensor,
        mask: Tensor,
        loss_normalization: Literal["sequence", "constant"] = "sequence",
        normalization_constant: int | None = None,
) -> Tensor:
    if loss_normalization == "sequence":
        seq_loss = (per_token_policy_gradient_loss * mask).sum(dim=-1) / mask.sum(dim=-1)
        return seq_loss.mean()
    elif loss_normalization == "constant":
        seq_loss = (per_token_policy_gradient_loss * mask).sum(dim=-1).sum()
        return seq_loss / normalization_constant
    else:
        raise ValueError(f"Invalid loss_normalization: {loss_normalization}. Must be 'sequence' or 'constant'.")