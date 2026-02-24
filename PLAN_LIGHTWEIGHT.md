# Lightweight Plan (Local / CPU)

Tasks that can be done without a GPU.

## Data Sources (from coursework spec)

- **Main TSV**: https://github.com/CRLala/NLPLabs-2024/blob/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv
- **Train/dev splits**: https://github.com/Perez-AlmendrosC/dontpatronizeme/tree/master/semeval-2022/practice%20splits
- **Test set**: https://github.com/Perez-AlmendrosC/dontpatronizeme/blob/master/semeval-2022/TEST/task4_test.tsv
- **PCL paper (Ex 1)**: https://aclanthology.org/2020.coling-main.518/
- **Baseline Colab**: https://colab.research.google.com/drive/1M5Qx-FVJYNqFdvpJgggIZaWk5SpGS1Nu

## Checklist

- [x] Repo setup — README, .gitignore, requirements.txt
- [x] Data download — `python data/download.py`
- [x] EDA script — `python notebooks/eda.py`
  - Technique 1: Class distribution + per-keyword PCL rates
  - Technique 2: Text length + TF-IDF bigram analysis
- [x] GPU plan — `PLAN_GPU.md`
- [x] Placeholders — `BestModel/train.py`, `BestModel/predict.py`

## What's Left (GPU-dependent)

See `PLAN_GPU.md` for model training, prediction, and evaluation tasks.
