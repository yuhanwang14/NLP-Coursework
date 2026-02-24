#!/usr/bin/env python3
"""
Train a binary PCL classifier by fine-tuning a pretrained transformer.

Approach:
    - Model: DeBERTa-v3-base (or RoBERTa-large)
    - Loss: CrossEntropyLoss with inverse class-frequency weights
    - Early stopping on internal validation F1

Usage:
    python BestModel/train.py [--model microsoft/deberta-v3-base]
                              [--lr 2e-5]
                              [--epochs 5]
                              [--batch_size 16]
                              [--max_len 256]
                              [--val_split 0.1]
                              [--patience 2]
                              [--output_dir BestModel/model]

See PLAN_GPU.md for full details.
"""

# TODO: Implement on GPU machine. See PLAN_GPU.md for the full approach.
#
# Skeleton:
#
# 1. Parse args (model name, lr, epochs, batch_size, max_len, val_split, patience, output_dir)
# 2. Load data/processed/train.csv
# 3. Stratified split → train / internal_val
# 4. Tokenize with AutoTokenizer
# 5. Compute class weights from training label distribution
# 6. Build model with AutoModelForSequenceClassification(num_labels=2)
# 7. Training loop:
#      - AdamW optimizer with weight decay
#      - Linear LR scheduler with warmup
#      - Evaluate F1 on internal_val each epoch
#      - Early stopping on val F1 (patience=2)
# 8. Save best checkpoint to output_dir
#
# Expected runtime: ~15-30 min per hyperparameter config on a single GPU.

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train PCL classifier")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--output_dir", default="BestModel/model")
    args = parser.parse_args()

    raise NotImplementedError(
        "This script must be run on a GPU machine. "
        "See PLAN_GPU.md for implementation details."
    )


if __name__ == "__main__":
    main()
