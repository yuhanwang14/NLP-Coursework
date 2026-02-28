#!/usr/bin/env python3
"""
Train a binary PCL classifier by fine-tuning a pretrained transformer.

Hybrid approach:
    - Moderate downsampling (configurable neg/pos ratio)
    - Mild class weighting in loss
    - Validate on INTERNAL dev split (held out from training data)
    - Final evaluation on official dev set (clean held-out result)

Usage:
    source .venv/bin/activate
    python BestModel/train.py --model roberta-large --batch_size 8 --epochs 8
"""

import argparse
import os
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from BestModel.dataset import PCLDataset, load_split


# ---------------------------------------------------------------------------
# Custom Trainer with class-weighted loss
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer that applies class weights to cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()  # DeBERTa fp16 safety
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights if self.class_weights is not None else None
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Compute F1, precision, recall for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1": f1_score(labels, preds, pos_label=1),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PCL classifier")
    parser.add_argument("--model", default="roberta-large")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output_dir", default="BestModel/model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg_ratio", type=float, default=3.0,
                        help="Ratio of negatives to positives after downsampling")
    parser.add_argument("--pos_weight", type=float, default=3.0,
                        help="Weight for positive class in loss function")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                        help="Fraction of training data held out for internal validation")
    parser.add_argument("--gradient_accumulation", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Model: {args.model}")

    # ---- Load training data ----
    full_train_df = load_split("data/processed/train.csv")
    print(f"Full train set: {len(full_train_df)} examples")

    # ---- Split into train_internal + val_internal (BEFORE downsampling) ----
    train_df, val_df = train_test_split(
        full_train_df,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=full_train_df["label"],
    )
    print(f"Internal train: {len(train_df)} (pos={train_df.label.sum()}, neg={len(train_df)-train_df.label.sum()})")
    print(f"Internal val:   {len(val_df)} (pos={val_df.label.sum()}, neg={len(val_df)-val_df.label.sum()})")

    # ---- Downsample negatives in train_internal only ----
    pos_df = train_df[train_df.label == 1]
    neg_df = train_df[train_df.label == 0]
    n_pos = len(pos_df)
    n_neg_keep = min(len(neg_df), int(n_pos * args.neg_ratio))
    neg_sample = neg_df.sample(n=n_neg_keep, random_state=args.seed)
    train_bal = (
        pos_df._append(neg_sample)
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )
    print(f"Train (downsampled): {len(train_bal)} (pos={n_pos}, neg={n_neg_keep}, ratio=1:{n_neg_keep/n_pos:.1f})")

    # ---- Tokenize ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = PCLDataset(train_bal["text"].tolist(), train_bal["label"].values.astype(int), tokenizer, args.max_len)
    val_ds = PCLDataset(val_df["text"].tolist(), val_df["label"].values.astype(int), tokenizer, args.max_len)

    # ---- Model ----
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    if "deberta" in args.model.lower():
        model.float()
        print("Applied float() for DeBERTa")

    # ---- Class weights ----
    class_weights = torch.tensor([1.0, args.pos_weight], dtype=torch.float)
    print(f"Class weights: {class_weights.tolist()}")

    # ---- Training (early stopping on INTERNAL val split) ----
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        seed=args.seed,
        fp16=False,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # Internal validation split, NOT the official dev set
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    # ---- Save ----
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---- Report internal val results ----
    val_results = trainer.evaluate()
    print(f"\nInternal val results: {val_results}")

    # ---- Final clean evaluation on OFFICIAL dev set ----
    dev_df = load_split("data/processed/dev.csv")
    dev_ds = PCLDataset(dev_df["text"].tolist(), dev_df["label"].values.astype(int), tokenizer, args.max_len)
    dev_results = trainer.evaluate(dev_ds)
    print(f"\n*** Official dev results (clean held-out): {dev_results}")
    print(f"*** Official Dev F1: {dev_results['eval_f1']:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
