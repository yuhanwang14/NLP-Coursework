#!/usr/bin/env python3
"""
Ablation study: isolate the contribution of each design choice.

Runs 5 variants, removing one improvement at a time:
  1. Full model (all improvements)
  2. roberta-base instead of roberta-large
  3. 2:1 downsampling instead of 5:1
  4. No class weighting (pos_weight=1.0)
  5. max_len=128 instead of 256

Each variant uses the same internal validation split (15% of train)
and reports results on both internal val and official dev.

Usage:
    python BestModel/ablation.py          # run all variants
    python BestModel/ablation.py --only 2 # run only variant 2

Estimated time: ~30 min per variant on a single GPU.
Total: ~2.5 hours for all 5 variants.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


# ---- Ablation configurations ----
VARIANTS = {
    "1_full": {
        "desc": "Full model (all improvements)",
        "args": {
            "--model": "roberta-large",
            "--neg_ratio": "5.0",
            "--pos_weight": "2.0",
            "--max_len": "256",
        },
    },
    "2_base_model": {
        "desc": "RoBERTa-base instead of large (isolates model size)",
        "args": {
            "--model": "roberta-base",
            "--neg_ratio": "5.0",
            "--pos_weight": "2.0",
            "--max_len": "256",
        },
    },
    "3_downsample_2to1": {
        "desc": "2:1 downsampling instead of 5:1 (isolates sampling strategy)",
        "args": {
            "--model": "roberta-large",
            "--neg_ratio": "2.0",
            "--pos_weight": "2.0",
            "--max_len": "256",
        },
    },
    "4_no_class_weight": {
        "desc": "No class weighting (isolates loss weighting)",
        "args": {
            "--model": "roberta-large",
            "--neg_ratio": "5.0",
            "--pos_weight": "1.0",
            "--max_len": "256",
        },
    },
    "5_short_context": {
        "desc": "max_len=128 instead of 256 (isolates context length)",
        "args": {
            "--model": "roberta-large",
            "--neg_ratio": "5.0",
            "--pos_weight": "2.0",
            "--max_len": "128",
        },
    },
}


def run_variant(name, config, common_args):
    """Run a single ablation variant by calling train.py as a subprocess."""
    output_dir = f"BestModel/ablation_models/{name}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "BestModel.train",
        "--output_dir", output_dir,
        "--epochs", "8",
        "--batch_size", str(common_args.batch_size),
        "--patience", "3",
        "--seed", "42",
        "--val_fraction", "0.15",
    ]

    # Add variant-specific args
    for flag, value in config["args"].items():
        cmd.extend([flag, value])

    print(f"\n{'='*70}")
    print(f"ABLATION VARIANT: {name}")
    print(f"Description: {config['desc']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    start = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (datetime.now() - start).total_seconds()

    return {
        "name": name,
        "desc": config["desc"],
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--only", type=int, default=None,
                        help="Run only variant N (1-5)")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    results = []
    variant_names = list(VARIANTS.keys())

    if args.only is not None:
        if 1 <= args.only <= 5:
            variant_names = [variant_names[args.only - 1]]
        else:
            print(f"Error: --only must be 1-5, got {args.only}")
            sys.exit(1)

    print(f"Running {len(variant_names)} ablation variant(s)...")

    for name in variant_names:
        config = VARIANTS[name]
        result = run_variant(name, config, args)
        results.append(result)

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    for r in results:
        status = "✓" if r["returncode"] == 0 else "✗"
        mins = r["elapsed_seconds"] / 60
        print(f"  {status} {r['name']:25s} — {mins:.1f} min — {r['desc']}")

    # Save results
    results_path = "BestModel/ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("\nNow run predict.py for each variant to get dev F1 scores,")
    print("or check the training output above for the Official Dev F1 lines.")


if __name__ == "__main__":
    main()
