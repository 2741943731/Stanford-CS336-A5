import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

# 2026.10.17

def tokenize_prompts(prompt_strs: list[str],
                    output_strs: list[str],
                    tokenizer: PreTrainedTokenizerBase
) -> dict[str, Tensor]:

    encoded_promots = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs]
    encoded_outputs = [tokenizer.encode(output, add_special_tokens=False) for output in output_strs]
    ids = [prompt + output for prompt, output in zip(encoded_promots, encoded_outputs)]
    max_len = max([len(ids[i]) for i in range(len(ids))])

    ids_padded = torch.full((len(ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((len(ids), max_len - 1), dtype=torch.long)
    for i, (prompt, id) in enumerate(zip(encoded_promots, ids)):
        ids_padded[i, :len(id)] = torch.tensor(id, dtype=torch.long)
        response_mask[i, max(len(prompt) - 1, 0):len(id) - 1] = 1

    input_ids = ids_padded[:, :-1]
    labels = ids_padded[:, 1:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
