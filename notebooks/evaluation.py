#!/usr/bin/env python3
"""
Error Analysis & Local Evaluation for Exercise 5.2 (5 marks).

Requires:
    - dev.txt (model predictions)
    - baseline_dev_preds.txt (baseline predictions from run_baseline.py)
    - BestModel/model/dev_logits.npy (raw logits from predict.py)
    - data/processed/dev.csv (ground truth)

Usage:
    python notebooks/evaluation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    classification_report,
)
from collections import Counter

from BestModel.dataset import load_predictions

FIG_DIR = "notebooks/figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ========================================================================
# 1. ERROR ANALYSIS (2.5 marks)
# ========================================================================
def error_analysis(dev_df, model_preds, baseline_preds):
    labels = dev_df["label"].values
    texts = dev_df["text"].values

    # ---- 1a. Confusion matrix ----
    cm = confusion_matrix(labels, model_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No PCL", "PCL"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix — DeBERTa-v3-base")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("✓ Confusion matrix saved")

    # ---- 1b. Sample FP and FN ----
    preds_arr = np.array(model_preds)
    fp_idx = np.where((preds_arr == 1) & (labels == 0))[0]
    fn_idx = np.where((preds_arr == 0) & (labels == 1))[0]

    np.random.seed(42)
    fp_sample = np.random.choice(fp_idx, min(20, len(fp_idx)), replace=False)
    fn_sample = np.random.choice(fn_idx, min(20, len(fn_idx)), replace=False)

    print(f"\n{'='*80}")
    print(f"FALSE POSITIVES ({len(fp_idx)} total, showing {len(fp_sample)})")
    print(f"{'='*80}")
    for i, idx in enumerate(fp_sample, 1):
        print(f"\n  FP-{i} (idx={idx}):")
        print(f"  {texts[idx][:300]}...")

    print(f"\n{'='*80}")
    print(f"FALSE NEGATIVES ({len(fn_idx)} total, showing {len(fn_sample)})")
    print(f"{'='*80}")
    for i, idx in enumerate(fn_sample, 1):
        print(f"\n  FN-{i} (idx={idx}):")
        print(f"  {texts[idx][:300]}...")

    # ---- 1c. Comparison with baseline ----
    both_correct = sum(
        1 for m, b, g in zip(model_preds, baseline_preds, labels)
        if m == g and b == g
    )
    both_wrong = sum(
        1 for m, b, g in zip(model_preds, baseline_preds, labels)
        if m != g and b != g
    )
    ours_right_base_wrong = sum(
        1 for m, b, g in zip(model_preds, baseline_preds, labels)
        if m == g and b != g
    )
    ours_wrong_base_right = sum(
        1 for m, b, g in zip(model_preds, baseline_preds, labels)
        if m != g and b == g
    )

    total = len(labels)
    print(f"\n{'='*80}")
    print("MODEL vs BASELINE COMPARISON")
    print(f"{'='*80}")
    print(f"  Both correct:                  {both_correct:4d} ({both_correct/total:.1%})")
    print(f"  Both wrong:                    {both_wrong:4d} ({both_wrong/total:.1%})")
    print(f"  Ours right, baseline wrong:    {ours_right_base_wrong:4d} ({ours_right_base_wrong/total:.1%})")
    print(f"  Ours wrong, baseline right:    {ours_wrong_base_right:4d} ({ours_wrong_base_right/total:.1%})")

    base_f1 = f1_score(labels, baseline_preds, pos_label=1)
    model_f1 = f1_score(labels, model_preds, pos_label=1)
    print(f"\n  Baseline F1: {base_f1:.4f}")
    print(f"  Model   F1: {model_f1:.4f}")
    print(f"  Δ F1:       {model_f1 - base_f1:+.4f}")

    # Comparison bar chart
    categories = ["Both\ncorrect", "Both\nwrong", "Ours right\nbase wrong", "Ours wrong\nbase right"]
    values = [both_correct, both_wrong, ours_right_base_wrong, ours_wrong_base_right]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#e67e22"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Model vs Baseline Prediction Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "model_vs_baseline.png"), dpi=150)
    plt.close(fig)
    print("✓ Model vs baseline chart saved")


# ========================================================================
# 2. LOCAL EVALUATION (2.5 marks)
# ========================================================================
def local_evaluation(dev_df, model_preds, logits):
    labels = dev_df["label"].values

    # ---- 2a. Precision-recall curve ----
    # Use softmax probability of class 1 as score
    probs = np.exp(logits[:, 1]) / np.exp(logits).sum(axis=1)
    precision, recall, thresholds = precision_recall_curve(labels, probs, pos_label=1)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#3498db", linewidth=2, label="PR Curve")
    ax.scatter([recall[best_idx]], [precision[best_idx]],
               color="red", s=100, zorder=5,
               label=f"Best F1={f1_scores[best_idx]:.3f} (thresh={best_threshold:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pr_curve.png"), dpi=150)
    plt.close(fig)

    # Predictions at optimal threshold
    optimal_preds = (probs >= best_threshold).astype(int)
    opt_f1 = f1_score(labels, optimal_preds, pos_label=1)
    default_f1 = f1_score(labels, model_preds, pos_label=1)
    print(f"\n{'='*80}")
    print("PRECISION-RECALL ANALYSIS")
    print(f"{'='*80}")
    print(f"  Default threshold (0.5) F1: {default_f1:.4f}")
    print(f"  Optimal threshold ({best_threshold:.3f}) F1: {opt_f1:.4f}")
    print(f"  Δ F1:                       {opt_f1 - default_f1:+.4f}")
    print("✓ PR curve saved")

    # ---- 2b. Per-keyword F1 ----
    if "keyword" in dev_df.columns:
        print(f"\n{'='*80}")
        print("PER-KEYWORD F1 BREAKDOWN")
        print(f"{'='*80}")
        keywords = dev_df["keyword"].unique()
        rows = []
        for kw in sorted(keywords):
            mask = dev_df["keyword"] == kw
            kw_labels = labels[mask]
            kw_preds = np.array(model_preds)[mask]
            n_pos = kw_labels.sum()
            n_total = len(kw_labels)
            if n_pos > 0:
                kw_f1 = f1_score(kw_labels, kw_preds, pos_label=1)
            else:
                kw_f1 = float("nan")
            rows.append({
                "keyword": kw,
                "total": n_total,
                "positive": int(n_pos),
                "pos_rate": n_pos / n_total,
                "F1": kw_f1,
            })

        kw_df = pd.DataFrame(rows).sort_values("F1", ascending=False)
        print(kw_df.to_string(index=False, float_format="%.3f"))

        # Per-keyword F1 bar chart
        kw_df_plot = kw_df.dropna(subset=["F1"]).sort_values("F1")
        fig, ax = plt.subplots(figsize=(10, max(5, len(kw_df_plot) * 0.5)))
        bars = ax.barh(kw_df_plot["keyword"], kw_df_plot["F1"],
                       color="#3498db", edgecolor="white")
        ax.set_xlabel("F1 Score")
        ax.set_title("Per-Keyword F1 Scores")
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, kw_df_plot["F1"]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "per_keyword_f1.png"), dpi=150)
        plt.close(fig)
        print("✓ Per-keyword F1 chart saved")


# ========================================================================
# Main
# ========================================================================
def main():
    dev_df = pd.read_csv("data/processed/dev.csv")
    model_preds = load_predictions("dev.txt")

    assert len(model_preds) == len(dev_df), (
        f"Mismatch: dev.txt has {len(model_preds)} lines, "
        f"dev.csv has {len(dev_df)} rows"
    )

    # ---- Error Analysis ----
    baseline_path = "baseline_dev_preds.txt"
    if os.path.exists(baseline_path):
        baseline_preds = load_predictions(baseline_path)
        assert len(baseline_preds) == len(dev_df)
        error_analysis(dev_df, model_preds, baseline_preds)
    else:
        print(f"⚠ {baseline_path} not found — skipping baseline comparison.")
        print("  Run: python BestModel/run_baseline.py")
        # Still do confusion matrix + FP/FN without baseline
        error_analysis(dev_df, model_preds, model_preds)

    # ---- Local Evaluation ----
    logits_path = "BestModel/model/dev_logits.npy"
    if os.path.exists(logits_path):
        logits = np.load(logits_path)
        local_evaluation(dev_df, model_preds, logits)
    else:
        print(f"\n⚠ {logits_path} not found — skipping PR curve.")
        print("  Run: python BestModel/predict.py (saves logits)")

    print(f"\n✅ All figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
