# PCL Binary Classification — SemEval 2022 Task 4, Subtask 1

Detecting Patronizing and Condescending Language (PCL) in news paragraphs.

- **Paper**: [Perez-Almendros et al. (2020)](https://aclanthology.org/2020.coling-main.518/)
- **Task**: [SemEval 2022 Task 4](https://sites.google.com/view/pcl-detection-semeval2022/)
- **Baseline Colab**: [RoBERTa-base baseline](https://colab.research.google.com/drive/1M5Qx-FVJYNqFdvpJgggIZaWk5SpGS1Nu)
- **EdStem**: [Course discussion](https://edstem.org/us/courses/86588/discussion)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare data
python data/download.py

# 3. Run EDA (Exercise 2)
python notebooks/eda.py

# 4. Train model on GPU machine (see PLAN_GPU.md)
python BestModel/train.py

# 5. Generate predictions
python BestModel/predict.py
```

## Repo Structure

```
├── data/
│   ├── download.py           # Download & prepare dataset
│   ├── raw/                   # Cloned repos (gitignored)
│   └── processed/             # Clean train/dev/test CSVs (gitignored)
├── notebooks/
│   ├── eda.py                 # Exploratory Data Analysis (Ex 2)
│   └── figures/               # EDA output plots
├── BestModel/
│   ├── train.py               # Model training (Ex 4)
│   ├── predict.py             # Inference → dev.txt, test.txt (Ex 5.1)
│   └── model/                 # Saved model weights (gitignored)
├── dev.txt                    # Predictions on official dev set
├── test.txt                   # Predictions on official test set
├── PLAN_LIGHTWEIGHT.md        # Local (CPU) tasks checklist
├── PLAN_GPU.md                # Remote (GPU) training guide
├── requirements.txt
└── README.md
```

## Task Summary

| Item | Details |
|------|---------|
| Input | News paragraph text |
| Output | `1` (PCL) or `0` (No PCL) |
| Metric | F1 score (positive class) |
| Baseline | RoBERTa-base → 0.48 dev / 0.49 test |
| Data | [Main TSV](https://github.com/CRLala/NLPLabs-2024/tree/main/Dont_Patronize_Me_Trainingset), [Splits](https://github.com/Perez-AlmendrosC/dontpatronizeme/tree/master/semeval-2022/practice%20splits), [Test](https://github.com/Perez-AlmendrosC/dontpatronizeme/blob/master/semeval-2022/TEST/task4_test.tsv) |

## Exercises (30 marks total)

| Ex | Topic | Marks | Time |
|----|-------|-------|------|
| 1 | Critical Paper Review | 6 | 3h |
| 2 | EDA (2 techniques) | 6 | 3h |
| 3 | Proposed Approach | 4 | 2h |
| 4 | Model Training + Repo | 1 | 8h |
| 5.1 | Global Eval (dev.txt, test.txt) | 6 | 30m |
| 5.2 | Local Eval (Error Analysis) | 5 | 3h |
| 6 | Report & Repo Quality | 2 | 30m |
