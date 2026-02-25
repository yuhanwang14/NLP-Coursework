# PCL Binary Classification — SemEval 2022 Task 4, Subtask 1

Detecting Patronizing and Condescending Language (PCL) in news paragraphs.

- **Paper**: [Perez-Almendros et al. (2020)](https://aclanthology.org/2020.coling-main.518/)
- **Task**: [SemEval 2022 Task 4](https://sites.google.com/view/pcl-detection-semeval2022/)
- **Baseline Colab**: [RoBERTa-base baseline](https://colab.research.google.com/drive/1M5Qx-FVJYNqFdvpJgggIZaWk5SpGS1Nu)
- **EdStem**: [Course discussion](https://edstem.org/us/courses/86588/discussion)

## Quick Start

```bash
# 1. Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download and prepare data
python data/download.py

# 3. Run EDA (Exercise 2)
python notebooks/eda.py

# 4. Reproduce baseline
python BestModel/run_baseline.py

# 5. Train model (GPU required, ~80 min on T4)
python BestModel/train.py

# 6. Generate predictions
python BestModel/predict.py

# 7. Run error analysis (Exercise 5.2)
python notebooks/evaluation.py
```

## Repo Structure

```
├── data/
│   ├── download.py              # Download & prepare dataset
│   ├── raw/                     # Cloned repos (gitignored)
│   └── processed/               # Clean train/dev/test CSVs (gitignored)
├── BestModel/
│   ├── __init__.py              # Package init
│   ├── dataset.py               # Shared Dataset classes & utils
│   ├── train.py                 # Model training (Ex 4)
│   ├── predict.py               # Inference → dev.txt, test.txt (Ex 5.1)
│   ├── run_baseline.py          # RoBERTa-base baseline reproduction
│   └── model/                   # Saved model weights (gitignored)
├── notebooks/
│   ├── eda.py                   # Exploratory Data Analysis (Ex 2)
│   ├── evaluation.py            # Error analysis & local eval (Ex 5.2)
│   └── figures/                 # Output plots
├── dev.txt                      # Predictions on official dev set
├── test.txt                     # Predictions on official test set
├── requirements.txt
└── README.md
```

## Task Summary

| Item | Details |
|------|---------| 
| Input | News paragraph text |
| Output | `1` (PCL) or `0` (No PCL) |
| Metric | F1 score (positive class) |
| Baseline | RoBERTa-base → 0.46 dev |
| **Our Model** | **RoBERTa-large → 0.575 dev** |
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
