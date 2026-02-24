#!/usr/bin/env python3
"""
Exploratory Data Analysis for the PCL Dataset.

Exercise 2: Two EDA techniques with visual evidence, analysis, and impact statement.

Technique 1 — Class Distribution Analysis
Technique 2 — Text Length & N-gram Analysis

Usage:
    python notebooks/eda.py

Outputs plots to notebooks/figures/
Requires: data/processed/train.csv (run data/download.py first)
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 12, "figure.figsize": (10, 6)})


def load_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run data/download.py first.")
        sys.exit(1)
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} training examples")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 1: Class Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def technique_1_class_distribution(df: pd.DataFrame):
    """Analyse and visualise the class imbalance in the PCL dataset."""
    print("\n" + "=" * 60)
    print("TECHNIQUE 1: Class Distribution Analysis")
    print("=" * 60)

    counts = df["label"].value_counts().sort_index()
    total = len(df)
    ratio = counts[0] / counts[1] if counts[1] > 0 else float("inf")

    print(f"\n  No PCL (0): {counts[0]:,}  ({counts[0]/total:.1%})")
    print(f"  PCL    (1): {counts[1]:,}  ({counts[1]/total:.1%})")
    print(f"  Imbalance ratio: {ratio:.1f}:1")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#4CAF50", "#F44336"]
    labels_map = {0: "No PCL", 1: "PCL"}

    # Absolute counts
    bars = axes[0].bar(
        [labels_map[i] for i in counts.index],
        counts.values,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{val:,}",
            ha="center",
            fontweight="bold",
            fontsize=13,
        )
    axes[0].set_title("Class Distribution (Counts)", fontweight="bold")
    axes[0].set_ylabel("Number of Samples")

    # Percentage pie
    axes[1].pie(
        counts.values,
        labels=[f"{labels_map[i]}\n({counts[i]/total:.1%})" for i in counts.index],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Class Proportion", fontweight="bold")

    fig.suptitle(f"PCL Dataset Class Distribution (Imbalance Ratio: {ratio:.1f}:1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: figures/class_distribution.png")

    # By keyword
    if "keyword" in df.columns:
        keyword_stats = df.groupby("keyword")["label"].agg(["count", "sum", "mean"])
        keyword_stats.columns = ["total", "pcl_count", "pcl_rate"]
        keyword_stats = keyword_stats.sort_values("pcl_rate", ascending=True)
        print("\n  PCL rate by keyword:")
        for kw, row in keyword_stats.iterrows():
            bar = "█" * int(row["pcl_rate"] * 40)
            print(f"    {kw:15s}  {row['pcl_rate']:.1%}  {bar}  (n={int(row['total'])})")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(keyword_stats.index, keyword_stats["pcl_rate"], color="#2196F3", edgecolor="white")
        ax.set_xlabel("PCL Rate")
        ax.set_title("PCL Rate by Keyword Group", fontweight="bold")
        ax.axvline(x=df["label"].mean(), color="red", linestyle="--", alpha=0.7, label=f"Overall: {df['label'].mean():.1%}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "pcl_rate_by_keyword.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → Saved: figures/pcl_rate_by_keyword.png")

    # Impact statement
    print("\n  IMPACT STATEMENT:")
    print(f"    The dataset is heavily imbalanced ({ratio:.1f}:1).")
    print("    This justifies using class-weighted loss during training")
    print("    and evaluating with F1 (not accuracy) as the primary metric.")


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 2: Text Length & N-gram Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def technique_2_length_and_ngrams(df: pd.DataFrame):
    """Analyse token-length distributions and discriminative n-grams per class."""
    print("\n" + "=" * 60)
    print("TECHNIQUE 2: Text Length & N-gram Analysis")
    print("=" * 60)

    # Word counts
    df = df.copy()
    df["word_count"] = df["text"].str.split().str.len()

    for label, name in [(0, "No PCL"), (1, "PCL")]:
        subset = df[df["label"] == label]["word_count"]
        print(f"\n  {name}: mean={subset.mean():.1f}, median={subset.median():.0f}, "
              f"std={subset.std():.1f}, min={subset.min()}, max={subset.max()}")

    # Box plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data_by_label = [df[df["label"] == l]["word_count"] for l in [0, 1]]
    bp = axes[0].boxplot(
        data_by_label,
        labels=["No PCL", "PCL"],
        patch_artist=True,
        boxprops=dict(linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], ["#4CAF50", "#F44336"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Word Count")
    axes[0].set_title("Text Length Distribution by Class", fontweight="bold")

    # Histogram overlay
    axes[1].hist(data_by_label[0], bins=50, alpha=0.6, label="No PCL", color="#4CAF50", density=True)
    axes[1].hist(data_by_label[1], bins=50, alpha=0.6, label="PCL", color="#F44336", density=True)
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Text Length Histogram", fontweight="bold")
    axes[1].legend()
    axes[1].set_xlim(0, 200)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "text_length.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: figures/text_length.png")

    # TF-IDF top n-grams per class
    print("\n  Top discriminative bigrams per class (by TF-IDF):")
    for label, name in [(0, "No PCL"), (1, "PCL")]:
        texts = df[df["label"] == label]["text"].tolist()
        tfidf = TfidfVectorizer(
            ngram_range=(2, 2),
            max_features=5000,
            stop_words="english",
            min_df=3,
        )
        matrix = tfidf.fit_transform(texts)
        mean_scores = matrix.mean(axis=0).A1
        top_idx = mean_scores.argsort()[-15:][::-1]
        features = tfidf.get_feature_names_out()
        print(f"\n    {name}:")
        for i, idx in enumerate(top_idx, 1):
            print(f"      {i:2d}. {features[idx]:30s} (tfidf={mean_scores[idx]:.4f})")

    # Top n-grams bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (label, name, color) in zip(axes, [(0, "No PCL", "#4CAF50"), (1, "PCL", "#F44336")]):
        texts = df[df["label"] == label]["text"].tolist()
        tfidf = TfidfVectorizer(ngram_range=(2, 2), max_features=5000, stop_words="english", min_df=3)
        matrix = tfidf.fit_transform(texts)
        mean_scores = matrix.mean(axis=0).A1
        top_idx = mean_scores.argsort()[-10:]
        features = tfidf.get_feature_names_out()
        ax.barh([features[i] for i in top_idx], mean_scores[top_idx], color=color, edgecolor="white")
        ax.set_title(f"Top Bigrams — {name}", fontweight="bold")
        ax.set_xlabel("Mean TF-IDF Score")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "top_bigrams.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: figures/top_bigrams.png")

    # Impact statement
    print("\n  IMPACT STATEMENT:")
    print("    PCL texts tend to be longer, suggesting max_seq_length=256 is appropriate.")
    print("    Discriminative bigrams reveal topical differences between classes,")
    print("    which can inform feature engineering or help debug model errors.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  PCL Dataset — Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()

    technique_1_class_distribution(df)
    technique_2_length_and_ngrams(df)

    print("\n" + "=" * 60)
    print("  EDA Complete! Figures saved to notebooks/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
