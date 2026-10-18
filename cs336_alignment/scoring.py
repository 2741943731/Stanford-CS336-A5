import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# 2026.10.17

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
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
    model.eval()
    result = {}
    with torch.no_grad():
        logits = model(input_ids).logits # b s v
        log_logits = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = torch.gather(log_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        result["log_probs"] = log_probs
        if return_token_entropy:
            token_entropy = -torch.sum(torch.exp(log_logits) * log_logits, dim=-1)
            result["token_entropy"] = token_entropy
            # -sum plogp 但是已经log_softmax了 所以要exp回去
    return result
        
