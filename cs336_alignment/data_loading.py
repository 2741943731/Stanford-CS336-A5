import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from pathlib import Path

class PackedSFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, seq_length: int, shuffle: bool = True):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []
        self.shuffle = shuffle

        documents = []
        template_path = Path("cs336_alignment/prompts_safety/alpaca_sft.prompt")
        template = template_path.read_text(encoding="utf-8").rstrip("\n")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                txt = json.loads(line)
                doc = template.format(
                    instruction=txt["prompt"],
                    response=txt["response"],
                )
                self.data.append(doc)

        if self.shuffle:
            random.shuffle(self.data)

        self.token_ids = []
        for doc in self.data:
            ids = self.tokenizer.encode(doc, add_special_tokens=True)
            self.token_ids.extend(ids)
            self.token_ids.append(self.tokenizer.eos_token_id)

    def __len__(self):
        return max(0, (len(self.token_ids) - 1) // self.seq_length)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        start = idx * self.seq_length
        input_ids = self.token_ids[start : start + self.seq_length]
        labels = self.token_ids[start + 1 : start + self.seq_length + 1]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }     


def iterate_batches(dataset: PackedSFTDataset, batch_size: int, shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
