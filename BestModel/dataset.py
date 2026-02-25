"""
Shared PyTorch Dataset classes and utility functions for PCL classification.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PCLDataset(Dataset):
    """
    Tokenized dataset for PCL classification (training / evaluation).

    Pre-tokenizes all texts at init for speed. Requires labels.
    """

    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class PCLInferenceDataset(Dataset):
    """
    Tokenized dataset for inference only (no labels).

    Tokenizes on-the-fly per item (more memory-friendly for large sets).
    """

    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def load_split(path):
    """Load a data split CSV, handling NaN text values."""
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("")
    return df


def write_predictions(preds, path):
    """Write predictions as one 0/1 per line."""
    with open(path, "w") as f:
        for p in preds:
            f.write(f"{int(p)}\n")


def load_predictions(path):
    """Load one-per-line 0/1 predictions."""
    with open(path) as f:
        return [int(line.strip()) for line in f]
