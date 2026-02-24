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

# TODO: Implement on GPU machine. See PLAN_GPU.md for details.
#
# Skeleton:
#
# 1. Load saved model and tokenizer from model_dir
# 2. Load data/processed/dev.csv and data/processed/test.csv
# 3. Tokenize and run inference (no grad)
# 4. Write predictions to dev.txt and test.txt in repo root
# 5. Print line counts for verification

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate PCL predictions")
    parser.add_argument("--model_dir", default="BestModel/model")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    raise NotImplementedError(
        "This script must be run on a GPU machine. "
        "See PLAN_GPU.md for implementation details."
    )


if __name__ == "__main__":
    main()
