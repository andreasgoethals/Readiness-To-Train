# Readiness to Train

**Prescriptive Analytics for Optimal Training Intensity**
KU Leuven & OH Leuven

---

## Overview

This project develops a **causal machine learning system** that estimates each OH Leuven player's daily *Readiness to Train* and recommends an optimal training intensity. The recommended intensity is a continuous score in **[0, 1]**: 0 means full rest, ~1 means match-equivalent effort.

The system is built within a **Dynamic Treatment Regime (DTR)** framework, using G-methods to handle the time-varying confounding that arises when coaching decisions are observational.

> **Important:** The objective is *not* injury minimisation. A policy that minimises injury risk trivially prescribes zero load. The true objective is **performance optimisation under biological constraints**.

---

## Why Causal Inference?

Standard ML asks *"what will happen?"*. This project asks *"what training intensity today best contributes to future match-day performance?"* Three reasons this requires causal reasoning:

1. **Time-Varying Confounding Affected by Prior Treatment** -- covariates at time *t* (fatigue, ACWR) are simultaneously *consequences* of treatment at *t-1* and *causes* of treatment at *t*. Classical regression fails here; G-methods are required.
2. **Treatment-Confounder Feedback** -- coaches prescribe hard sessions when players look fresh. The observed load-performance association conflates physiological response with coach selection behaviour.
3. **Sequential Decision-Making** -- the question is not just *"what training today?"* but *"what sequence of intensities over the match cycle maximises match-day performance?"* -- a DTR problem.

---

## Project Structure

```
Readiness-To-Train/
├── data/
│   ├── raw/                          # Original xlsx files (NDA-protected, gitignored)
│   │   ├── Readiness_Data.xlsx       # 14,359 rows x 24 cols, 28 players (daily)
│   │   ├── Raw_Data.xlsx             # 9,968 rows x 39 cols, 84 players (session-level)
│   │   ├── Sessions.xlsx             # 1,206 rows x 9 cols (team-session metadata)
│   │   └── Games.xlsx                # 403 rows x 9 cols, 24 players (match performance)
│   └── processed/
│       └── RTT.xlsx                  # Auto-generated merged dataset (14,359 x 46 cols)
│
├── notebooks/
│   ├── 0. Processed_Data_Quality.ipynb     # Data quality: automated checks on RTT.xlsx
│   ├── 0. TI_Missingness_Analysis.ipynb    # Data quality: Training Intensity NaN patterns
│   ├── 1.1. Match Analysis.ipynb           # EDA: match-level data exploration
│   ├── 1.2. Raw Data Visualisation.ipynb   # EDA: all 4 raw datasets
│   ├── 1.3. Processed Data Visualisation.ipynb  # EDA: processed RTT.xlsx
│   ├── 2.1. Experiment1.ipynb              # Experiment 1: Match Intensity prediction
│   ├── 2.2. Experiment2.ipynb              # Experiment 2: Training Intensity prediction
│   └── 2.3. Experiment3.ipynb              # Experiment 3: Status Decrease prediction
│
├── scripts/
│   ├── Experiment1.py                # Experiment 1 runner
│   ├── Experiment2.py                # Experiment 2 runner
│   ├── Experiment3.py                # Experiment 3 runner
│   ├── generate_visualizations.py    # Batch DAG generation for all 28 players
│   └── generate_project_results.py   # Generate Project Results.pdf
│
├── src/
│   ├── data/
│   │   ├── data_preprocessing.py     # Multi-dataset merge & feature engineering
│   │   └── data_loader.py            # ML-ready dataset creation with lag & splits
│   ├── methods/
│   │   └── dag_creator.py            # Causal DAG builder (longitudinal, player-specific)
│   ├── models/
│   │   ├── lin_reg.py                # Ridge Regression (HPO via Optuna)
│   │   ├── log_reg.py                # Logistic Regression (HPO via Optuna)
│   │   ├── xgboost.py                # XGBoost (GPU-accelerated, early stopping)
│   │   ├── catboost.py               # CatBoost (GPU, native categoricals, HPO)
│   │   └── tabpfn.py                 # TabPFN v2 (in-context learning, GPU)
│   └── utils/
│       ├── generate_project_overview.py   # Generate Project Overview.pdf
│       └── generate_raw_data_dict.py      # Generate Raw Data Dictionary.pdf
│
├── images/DAGs/                      # Per-player causal DAG visualizations
├── results/                          # Model outputs
├── Project Overview.pdf              # Problem statement & research design
├── Project Results.pdf               # Experimental findings & conclusions
├── CLAUDE.md                         # Full technical documentation
├── requirements.txt
└── .gitignore
```

---

## Notebooks

### Data Quality Checks (prefix `0.`)

| Notebook | Description |
|----------|-------------|
| `0. Processed_Data_Quality` | Automated quality checks on the processed RTT.xlsx: player coverage, column completeness, temporal integrity, ACWR flag validation |
| `0. TI_Missingness_Analysis` | Investigation of missing values in Training Intensity Yesterday: why certain rows have NaN, validation of the free-day fill logic |

### Data Visualisation & EDA (prefix `1.x`)

| Notebook | Description |
|----------|-------------|
| `1.1. Match Analysis` | Match-level data exploration from Games.xlsx: match intensity distributions, per-player performance profiles, playing time patterns |
| `1.2. Raw Data Visualisation` | Comprehensive EDA across all 4 raw datasets: missingness heatmaps, dataset linkage (Venn diagrams), temporal coverage, per-player radar charts |
| `1.3. Processed Data Visualisation` | EDA of the processed RTT.xlsx: variable distributions, temporal patterns, per-player wellness trajectories, ACWR time series, correlation heatmaps |

### Experiments (prefix `2.x`)

| Notebook | Target | Task | Description |
|----------|--------|------|-------------|
| `2.1. Experiment 1` | Match Intensity Yesterday | Regression | Attempts to predict match-day performance from morning covariates. Serves as a reference to test whether match intensity is predictable at all. Includes full model comparison (Ridge, XGBoost, CatBoost, TabPFN) across 10 lag values, feature importance analysis, and per-player breakdown |
| `2.2. Experiment 2` | Training Intensity Yesterday | Regression | Models the coaching staff's load-assignment decisions from morning player state. Compares 4 models across 3 lag values with SHAP / permutation importance analysis. The fitted model can serve as a propensity model for downstream causal estimation |
| `2.3. Experiment 3` | Status Decrease | Classification | Predicts next-day player status deterioration (binary, ~3-5% prevalence). Runs in two modes: pure prediction (early-warning) and causal framing (adds training intensity as a covariate). Includes ROC/PR analysis and per-player AUC breakdown |

---

## Data Pipeline

```
Readiness_Data.xlsx  ---+
Raw_Data.xlsx        ---+-- data_preprocessing.py --> data/processed/RTT.xlsx
Sessions.xlsx        ---+                             (14,359 rows x 46 cols)
Games.xlsx           ---+
```

The processed dataset merges all four raw sources:
- **Readiness_Data** (base): daily player monitoring -- wellness z-scores, GPS benchmarks, ACWR, medical status
- **Raw_Data** (merged): detailed session-level GPS/HR, shifted +1 day ("Yesterday" columns)
- **Sessions** (merged): team-level session metadata for match day detection
- **Games** (merged): match performance data, shifted +1 day

All preprocessing is fully reproducible from the raw files via `python src/data/data_preprocessing.py`.

---

## Causal Framework

| Variable | Symbol | Examples |
|----------|--------|---------|
| Player covariates | **L_t** | Wellness z-scores, ACWR, GPS %, Days Since/Until Match |
| Treatment | **A_t** | Training Intensity in [0,1) |
| Outcome | **Y** | Status Decrease (short-term), Match Intensity (long-term) |

**Causal methods:** G-computation, MSM/IPTW, G-Estimation/SNMM, Q-learning, dWOLS

---

## Quick Start

```bash
# 1. Install dependencies (PyTorch CUDA must be installed separately)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Regenerate processed data
python src/data/data_preprocessing.py

# 3. Run experiments
python scripts/Experiment1.py
python scripts/Experiment2.py
python scripts/Experiment3.py

# 4. Generate reports
python scripts/generate_project_results.py
```

---

## GPU Acceleration

All GPU-capable models run on CUDA by default:

| Model | GPU parameter | Requirement |
|-------|--------------|-------------|
| XGBoost | `device='cuda'` | XGBoost >= 2.0, CUDA toolkit |
| CatBoost | `task_type='GPU'` | CatBoost >= 1.2, CUDA toolkit |
| TabPFN | `device='cuda'` | PyTorch **CUDA build**, tabpfn >= 2.0 |

**Important:** `pip install torch` installs CPU-only by default. Install the CUDA build first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## Generated Documents

| Document | Generator | Description |
|----------|-----------|-------------|
| `Project Overview.pdf` | `src/utils/generate_project_overview.py` | Problem statement, research design, causal framework |
| `Project Results.pdf` | `scripts/generate_project_results.py` | Experimental findings and conclusions |
| `data/raw/Raw Data Dictionary.pdf` | `src/utils/generate_raw_data_dict.py` | Documentation of all raw data columns |
| `data/processed/RTT Data Dictionary.pdf` | `src/data/data_preprocessing.py` | Documentation of processed dataset columns |

---

## Key Dependencies

```
torch >= 2.0 (CUDA)   pandas >= 2.0        numpy >= 1.24       scipy >= 1.10
scikit-learn >= 1.3    xgboost >= 2.0       catboost >= 1.2     optuna >= 4.0
tabpfn >= 2.0          shap >= 0.44         tqdm >= 4.0
networkx >= 3.0        matplotlib >= 3.7    seaborn >= 0.12
reportlab >= 4.0       openpyxl >= 3.1      ipywidgets >= 8.0
```

---

## References

- Chakraborty & Moodie (2013). *Statistical Methods for Dynamic Treatment Regimes*. Springer.
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Gabbett (2016). The training-injury prevention paradox. *BJSM, 50*(5), 273-280.
- Hernan & Robins (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Murphy (2003). Optimal dynamic treatment regimes. *JRSS-B, 65*(2), 331-355.
- Robins (1986). A new approach to causal inference in mortality studies. *Mathematical Modelling, 7*, 1393-1512.
- Wallace & Moodie (2015). Doubly-robust DTR estimation via weighted least squares. *Biometrics, 71*(3), 636-644.

---

*KU Leuven & OH Leuven -- March 2026*
