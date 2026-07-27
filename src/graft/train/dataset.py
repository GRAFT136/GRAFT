
from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

class NNPJsonlDataset(Dataset):

    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int = 512,
        chat_template: bool = True,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template = chat_template

    @classmethod
    def from_jsonl(cls, path: str, tokenizer, max_length: int = 512, chat_template: bool = True):
        records: List[Dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records, tokenizer, max_length, chat_template)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        query = rec["query"]
        answer = rec["answer"]

        if self.chat_template:
            text = (
                f"<|im_start|>user\n{query}<|im_end|>\n"
                f"<|im_start|>assistant\n{answer}<|im_end|>"
            )
        else:
            text = f"Q: {query}\nA: {answer}"

        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        assistant_token = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        labels = _mask_prompt(labels, assistant_token)

        item = {"input_ids": input_ids, "labels": labels}
        if rec.get("community") is not None:
            item["community"] = torch.tensor(int(rec["community"]), dtype=torch.long)
        return item

def _mask_prompt(labels: torch.Tensor, assistant_token_ids: List[int]) -> torch.Tensor:
    n = len(assistant_token_ids)
    ids = labels.tolist()
    for i in range(len(ids) - n):
        if ids[i : i + n] == assistant_token_ids:
            labels[: i + n] = -100
            break
    return labels

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attention_mask[i, :L] = 1
    out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    if "community" in batch[0]:
        out["community"] = torch.stack([b["community"] for b in batch])
    return out
