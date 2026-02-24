#!/usr/bin/env python3
"""
Download and prepare the Don't Patronize Me! PCL dataset.

Sources (from coursework spec):
  - Main TSV:      https://github.com/CRLala/NLPLabs-2024/.../dontpatronizeme_pcl.tsv
  - Train/dev IDs: https://github.com/Perez-AlmendrosC/dontpatronizeme/.../practice splits/
  - Test set:      https://github.com/Perez-AlmendrosC/dontpatronizeme/.../TEST/task4_test.tsv

Usage:
    python data/download.py
"""

import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed")

# Two repos to clone
REPOS = {
    "nlplabs": {
        "url": "https://github.com/CRLala/NLPLabs-2024.git",
        "dir": os.path.join(RAW_DIR, "NLPLabs-2024"),
    },
    "dpm": {
        "url": "https://github.com/Perez-AlmendrosC/dontpatronizeme.git",
        "dir": os.path.join(RAW_DIR, "dontpatronizeme"),
    },
}


def clone_repo(name: str, url: str, dest: str):
    if os.path.exists(dest):
        print(f"  ✓ {name} already cloned")
        return
    print(f"  ↓ Cloning {name}...")
    subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)


def load_main_tsv(path: str) -> pd.DataFrame:
    """
    Load dontpatronizeme_pcl.tsv (skip 4 header lines).
    Binary labels: 0/1 → 0 (No PCL), 2/3/4 → 1 (PCL).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[4:]:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            orig = parts[-1]
            rows.append({
                "par_id": parts[0],
                "art_id": parts[1],
                "keyword": parts[2],
                "country": parts[3],
                "text": parts[4],
                "label": 0 if orig in ("0", "1") else 1,
                "orig_label": orig,
            })
    return pd.DataFrame(rows)


def load_split_ids(path: str) -> set:
    """Load paragraph IDs from a practice split CSV."""
    df = pd.read_csv(path)
    return set(df["par_id"].astype(str).tolist())


def load_test_set(path: str) -> pd.DataFrame:
    """Load official test set TSV (no labels)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                rows.append({
                    "par_id": parts[0],
                    "art_id": parts[1],
                    "keyword": parts[2],
                    "country": parts[3],
                    "text": parts[4],
                })
    return pd.DataFrame(rows)


def main():
    print("=== PCL Dataset Download & Preparation ===\n")
    os.makedirs(RAW_DIR, exist_ok=True)

    # Step 1: Clone repos
    print("Step 1: Cloning repos...")
    for name, info in REPOS.items():
        clone_repo(name, info["url"], info["dir"])

    # Locate files
    main_tsv = os.path.join(
        REPOS["nlplabs"]["dir"],
        "Dont_Patronize_Me_Trainingset",
        "dontpatronizeme_pcl.tsv",
    )
    splits_dir = os.path.join(
        REPOS["dpm"]["dir"], "semeval-2022", "practice splits"
    )
    train_ids_path = os.path.join(splits_dir, "train_semeval_parids-labels.csv")
    dev_ids_path = os.path.join(splits_dir, "dev_semeval_parids-labels.csv")
    test_path = os.path.join(
        REPOS["dpm"]["dir"], "semeval-2022", "TEST", "task4_test.tsv"
    )

    # Verify files exist
    for name, path in [("Main TSV", main_tsv), ("Train IDs", train_ids_path),
                        ("Dev IDs", dev_ids_path), ("Test set", test_path)]:
        if not os.path.exists(path):
            print(f"  ✗ Missing: {name} at {path}")
            sys.exit(1)
        print(f"  ✓ Found: {name}")

    # Step 2: Build splits
    print("\nStep 2: Building clean splits...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_data = load_main_tsv(main_tsv)
    print(f"  Total paragraphs: {len(all_data)}")
    print(f"  Class distribution: {dict(all_data['label'].value_counts())}")

    train_ids = load_split_ids(train_ids_path)
    dev_ids = load_split_ids(dev_ids_path)

    train_df = all_data[all_data["par_id"].isin(train_ids)].copy()
    dev_df = all_data[all_data["par_id"].isin(dev_ids)].copy()
    test_df = load_test_set(test_path)

    # Save
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(PROCESSED_DIR, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"\n  → train.csv: {len(train_df)} rows (PCL={int(train_df['label'].sum())}, No_PCL={len(train_df)-int(train_df['label'].sum())})")
    print(f"  → dev.csv:   {len(dev_df)} rows (PCL={int(dev_df['label'].sum())}, No_PCL={len(dev_df)-int(dev_df['label'].sum())})")
    print(f"  → test.csv:  {len(test_df)} rows (no labels)")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
