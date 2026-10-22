import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# 2026.10.17

# Chunk size (in sequence positions) for the entropy computation. Entropy is a
# logging metric, but the naive `-(p * log p).sum(-1)` allocates two extra
# (batch, seq, vocab) tensors at full size, which is easily several GiB.
_ENTROPY_CHUNK = 64


def _token_entropy(log_logits: Tensor) -> Tensor:
    """Per-token entropy over the vocabulary, chunked and without autograd."""
    pieces = []
    with torch.no_grad():
        for start in range(0, log_logits.shape[1], _ENTROPY_CHUNK):
            chunk = log_logits[:, start : start + _ENTROPY_CHUNK, :].detach()
            pieces.append(-(chunk.exp() * chunk).sum(dim=-1))
    return torch.cat(pieces, dim=1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
    need_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    if not need_grad:
        with torch.no_grad():
            logits = model(input_ids).logits  # b s v
            log_logits = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs = torch.gather(log_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            result = {"log_probs": log_probs}
            if return_token_entropy:
                result["token_entropy"] = _token_entropy(log_logits)
            return result

    result = {}
    logits = model(input_ids).logits  # b s v
    log_logits = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result["log_probs"] = log_probs
    if return_token_entropy:
        # -sum p log p, where p = exp(log_logits); see _token_entropy for why
        # this is chunked and detached from the autograd graph.
        result["token_entropy"] = _token_entropy(log_logits)
    return result
