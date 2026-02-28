# Natural Language Processing Coursework 2026

**Yuhan Wang** | Leaderboard: `YuhanWang`

## Setup

```bash
git clone https://github.com/yuhanwang14/NLP-Coursework.git
cd NLP-Coursework
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch transformers datasets scikit-learn
python data/download.py
```

## Reproduce

### Exercise 2 — EDA

```bash
python notebooks/eda.py
# → notebooks/figures/class_distribution.png
# → notebooks/figures/pcl_rate_by_keyword.png
# → notebooks/figures/text_length.png
# → notebooks/figures/top_bigrams.png
```

### Exercise 4 — Model Training (GPU required)

**Baseline reproduction** (~15 min on T4):
```bash
python BestModel/run_baseline.py
# → baseline_dev_preds.txt  (expected F1 ≈ 0.46)
```

**Our model** — fine-tune RoBERTa-large (~80 min on T4):
```bash
python BestModel/train.py \
    --model roberta-large \
    --batch_size 8 \
    --epochs 8 \
    --lr 2e-5 \
    --neg_ratio 5.0 \
    --pos_weight 2.0 \
    --patience 3
# → BestModel/model/  (saved checkpoint)
```

### Exercise 5.1 — Generate Predictions

```bash
python BestModel/predict.py
# → dev.txt   (2,094 predictions, F1 = 0.5747)
# → test.txt  (3,832 predictions)
```

### Exercise 5.2 — Error Analysis

```bash
python notebooks/evaluation.py
# → notebooks/figures/confusion_matrix.png
# → notebooks/figures/model_vs_baseline.png
# → notebooks/figures/pr_curve.png
# → notebooks/figures/per_keyword_f1.png
```

## Repo Structure

```
├── BestModel/
│   ├── train.py             # Model training
│   ├── predict.py           # Inference → dev.txt, test.txt
│   ├── run_baseline.py      # RoBERTa-base baseline
│   ├── dataset.py           # Dataset classes & utils
│   └── model/               # Saved weights (gitignored)
├── data/
│   └── download.py          # Download & prepare dataset
├── notebooks/
│   ├── eda.py               # Exploratory Data Analysis
│   ├── evaluation.py        # Error analysis & local eval
│   └── figures/             # Output plots
├── LaTeX/
│   ├── report.tex           # Coursework report
│   └── references.bib       # BibTeX references
├── dev.txt                  # Dev set predictions
├── test.txt                 # Test set predictions
└── baseline_dev_preds.txt   # Baseline predictions
```

## Results

| Model | Dev F1 |
|-------|--------|
| Baseline (RoBERTa-base) | 0.460 |
| **Ours (RoBERTa-large)** | **0.5747** |
