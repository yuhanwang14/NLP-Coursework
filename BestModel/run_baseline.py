#!/usr/bin/env python3
"""
Reproduce the RoBERTa-base baseline from the course Colab notebook.

Mirrors the Colab approach:
  1. Downsample negatives to 2× positives
  2. Fine-tune roberta-base for 1 epoch
  3. Save dev predictions → baseline_dev_preds.txt

Expected dev F1 ≈ 0.48

Usage:
    python BestModel/run_baseline.py
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score
from collections import Counter

from BestModel.dataset import PCLDataset, load_split, write_predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load data ----
    train_df = load_split("data/processed/train.csv")
    dev_df = load_split("data/processed/dev.csv")
    print(f"Train: {len(train_df)}  Dev: {len(dev_df)}")

    # ---- Downsample negatives (2× positives, matching Colab) ----
    pos = train_df[train_df.label == 1]
    neg = train_df[train_df.label == 0].sample(n=len(pos) * 2, random_state=42)
    training_set = pos._append(neg).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Downsampled: {len(training_set)}  (pos={len(pos)}, neg={len(neg)})")

    # ---- Tokenize ----
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = PCLDataset(
        training_set["text"].tolist(),
        training_set["label"].values.astype(int),
        tokenizer, max_len=128,
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader),
    )

    # ---- Train 1 epoch (matching Colab) ----
    model.train()
    for step, batch in enumerate(train_loader, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 20 == 0:
            print(f"  Step {step}/{len(train_loader)}  loss={outputs.loss.item():.4f}")

    # ---- Predict on dev ----
    model.eval()
    dev_ds = PCLDataset(
        dev_df["text"].tolist(),
        dev_df["label"].values.astype(int),
        tokenizer, max_len=128,
    )
    dev_loader = DataLoader(dev_ds, batch_size=32)

    preds = []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    print(f"Prediction distribution: {Counter(preds)}")

    # ---- Save ----
    write_predictions(preds, "baseline_dev_preds.txt")
    f1 = f1_score(dev_df["label"].values, preds, pos_label=1)
    print(f"\nBaseline Dev F1: {f1:.4f}")
    print("Saved: baseline_dev_preds.txt")


if __name__ == "__main__":
    main()
