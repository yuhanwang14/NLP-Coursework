# GPU Plan (Remote Machine)

Step-by-step guide for model training and evaluation on a GPU machine.

## Prerequisites

```bash
git clone <your-repo-url>
cd NLP
pip install -r requirements.txt
pip install torch transformers datasets
python data/download.py   # if data not already present
```

## Step 1 — Reproduce the Baseline

Before building your own model, reproduce the RoBERTa-base baseline:
- **Baseline Colab**: https://colab.research.google.com/drive/1M5Qx-FVJYNqFdvpJgggIZaWk5SpGS1Nu
- Expected: F1 ≈ 0.48 on dev set

This gives you baseline predictions to compare against in error analysis (Ex 5.2).

## Step 2 — Train Your Model

```bash
python BestModel/train.py
```

### Approach: Fine-tune DeBERTa-v3-base

- **Model**: `microsoft/deberta-v3-base` (or try `roberta-large`)
- **Loss**: `CrossEntropyLoss` with class weights inversely proportional to class frequency
- **Hyperparameters**:
  | Param | Values to try |
  |-------|--------------|
  | Learning rate | `1e-5`, `2e-5`, `3e-5` |
  | Epochs | 3, 4, 5 |
  | Batch size | 16 |
  | Max seq length | 256 |
- **Validation**: Stratified 90/10 split from official training set
- **Early stopping**: On internal val F1 (patience=2)
- **Save**: Best checkpoint → `BestModel/model/`

### Optional Enhancements

1. Data augmentation (back-translation, synonym replacement)
2. Ensemble: average predictions from top-2 checkpoints
3. Try `deberta-v3-large` if VRAM allows
4. Layer-wise learning rate decay
5. Use the 7-category labels as auxiliary features (from dontpatronizeme_categories.tsv)

## Step 3 — Generate Predictions (Exercise 5.1)

```bash
python BestModel/predict.py
```

- Produces `dev.txt` and `test.txt` in repo root
- Format: one `0` or `1` per line, matching input order
- Verify line counts: `wc -l dev.txt test.txt`

### Quick F1 Check

```bash
python -c "
from sklearn.metrics import f1_score
import pandas as pd
dev = pd.read_csv('data/processed/dev.csv')
preds = [int(l.strip()) for l in open('dev.txt')]
print(f'Dev F1: {f1_score(dev.label, preds):.4f}')
"
```

Targets:
- Dev F1 > 0.48 (1 mark), Test F1 > 0.49 (1 mark)
- Top 60% → +1, Top 30% → +1 more, Top 10% → +1 more

## Step 4 — Error Analysis & Ablations (Exercise 5.2 — 5 marks)

Create `notebooks/evaluation.py`:

### Error Analysis (2.5 marks)
1. Confusion matrix on dev set
2. Sample 20 FP and 20 FN, categorise failure modes
3. Compare with baseline predictions:
   - Both correct, both wrong, yours right/baseline wrong, yours wrong/baseline right

### Other Local Evaluation (2.5 marks)
1. Ablation: with vs without class weighting
2. Precision-recall curve, find optimal threshold
3. Per-keyword F1 breakdown (disabled, homeless, immigrant, etc.)

## Step 5 — Report (Exercises 1, 2, 3, 5.2, 6)

**Sections** (submit as PDF):
1. **Ex 1**: Critical review of [PCL paper](https://aclanthology.org/2020.coling-main.518/) — Q1 contributions, Q2 strengths, Q3 weaknesses
2. **Ex 2**: EDA — embed plots from `notebooks/figures/`
3. **Ex 3**: Proposed approach — describe BestModel, rationale + expected outcome
4. **Ex 4/5.1**: Link to this repo (front page)
5. **Ex 5.2**: Error analysis figures/tables
6. **Ex 6**: Overall quality of report and repo

**Front page must include**:
- Clickable link to your GitHub/GitLab repo
- Leaderboard name (up to 20 chars)
