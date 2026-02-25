#!/usr/bin/env python3
"""
Generate predictions using the trained PCL classifier.

Produces:
    - dev.txt  — predictions on official dev set
    - test.txt — predictions on official test set

Format: one 0 or 1 per line, matching input order.

Usage:
    python BestModel/predict.py [--model_dir BestModel/model]
                                 [--max_len 256]
                                 [--batch_size 32]
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score

from BestModel.dataset import PCLInferenceDataset, load_split, write_predictions


def predict(model, loader, device):
    """Run inference and return predicted labels + raw logits."""
    model.eval()
    all_preds, all_logits = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    return all_preds, np.array(all_logits)


def main():
    parser = argparse.ArgumentParser(description="Generate PCL predictions")
    parser.add_argument("--model_dir", default="BestModel/model")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    # ---- Dev set ----
    dev_df = load_split("data/processed/dev.csv")
    dev_ds = PCLInferenceDataset(dev_df["text"].tolist(), tokenizer, args.max_len)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size)

    dev_preds, dev_logits = predict(model, dev_loader, device)
    write_predictions(dev_preds, "dev.txt")
    np.save("BestModel/model/dev_logits.npy", dev_logits)

    if "label" in dev_df.columns:
        print(f"Dev F1: {f1_score(dev_df['label'].values, dev_preds, pos_label=1):.4f}")

    # ---- Test set ----
    test_df = load_split("data/processed/test.csv")
    test_ds = PCLInferenceDataset(test_df["text"].tolist(), tokenizer, args.max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    test_preds, test_logits = predict(model, test_loader, device)
    write_predictions(test_preds, "test.txt")
    np.save("BestModel/model/test_logits.npy", test_logits)

    # ---- Summary ----
    print(f"\ndev.txt:  {len(dev_preds)} lines  (expected {len(dev_df)})")
    print(f"test.txt: {len(test_preds)} lines  (expected {len(test_df)})")
    print(f"Dev  pos rate: {sum(dev_preds)}/{len(dev_preds)} = {sum(dev_preds)/len(dev_preds):.3f}")
    print(f"Test pos rate: {sum(test_preds)}/{len(test_preds)} = {sum(test_preds)/len(test_preds):.3f}")


if __name__ == "__main__":
    main()
