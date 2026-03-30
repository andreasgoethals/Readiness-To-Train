# Readiness to Train

**Prescriptive Analytics for Optimal Training Intensity**
KU Leuven & OH Leuven — PhD Research Project

---

## Overview

This project develops a **causal machine learning system** that estimates each OH Leuven player's daily *Readiness to Train* and recommends an optimal training intensity. The recommended intensity is a continuous score in **[0, 1]**: 0 means full rest, ~1 means match-equivalent effort.

The system is built within a **Dynamic Treatment Regime (DTR)** framework, using G-methods to handle the time-varying confounding that arises when coaching decisions are observational. The research is conducted as part of a PhD project at KU Leuven in partnership with OH Leuven professional football club.

> **Important:** The objective is *not* injury minimisation. A policy that minimises injury risk trivially prescribes zero load. The true objective is **performance optimisation under biological constraints**.

---

## Why Causal Inference?

Standard ML asks *"what will happen?"*. This project asks *"what training intensity today best contributes to future match-day performance?"* Three reasons this requires causal reasoning:

1. **Time-Varying Confounding Affected by Prior Treatment** — covariates at time *t* (fatigue, ACWR) are simultaneously *consequences* of treatment at *t−1* and *causes* of treatment at *t*. Classical regression fails here; G-methods are required.
2. **Treatment-Confounder Feedback** — coaches prescribe hard sessions when players look fresh. The observed load-performance association conflates physiological response with coach selection behaviour.
3. **Sequential Decision-Making** — the question is not just *"what training today?"* but *"what sequence of intensities over the match cycle maximises match-day performance?"* — a DTR problem.

---

## Project Structure

```
Readiness-To-Train/
├── data/
│   ├── raw/                     # Original xlsx files (NDA-protected, gitignored)
│   │   ├── Readiness_Data.xlsx  # 14,359 rows × 24 cols, 28 players (daily)
│   │   ├── Raw_Data.xlsx        # 9,968 rows × 39 cols, 84 players (session-level)
│   │   ├── Sessions.xlsx        # 1,206 rows × 9 cols (team-session)
│   │   └── Games.xlsx           # 403 rows × 9 cols, 24 players (match-level)
│   └── processed/
│       └── RTT.xlsx             # Auto-generated merged dataset (14,359 × 46 cols)
│
├── notebooks/
│   ├── 0. Outlier_Detection.ipynb
│   ├── 0. Processed_Data_Quality.ipynb
│   ├── 0. TI_Missingness_Analysis.ipynb
│   ├── 1.1. Raw Data Visualisation.ipynb
│   ├── 1.2. Processed Data Visualisation.ipynb
│   ├── 2.1. Experiment1.ipynb   # Exp 1: Match Intensity (reference only)
│   ├── 2.2. Experiment2.ipynb   # Exp 2: Treatment policy (propensity model)
│   └── 2.3. Experiment3.ipynb   # Exp 3: Short-term load response (outcome model)
│
├── scripts/
│   ├── Experiment1.py           # Predict Match Intensity (reference baseline)
│   ├── Experiment2.py           # Predict Training Intensity (propensity model)
│   ├── Experiment3.py           # Predict Status Decrease (outcome model)
│   └── generate_visualizations.py  # Batch DAG generation for all 28 players
│
├── src/
│   ├── data/
│   │   ├── data_preprocessing.py  # Multi-dataset merge & feature engineering
│   │   └── data_loader.py         # ML-ready dataset creation with lag & splits
│   ├── methods/
│   │   └── dag_creator.py         # Causal DAG builder (longitudinal, player-specific)
│   ├── models/
│   │   ├── lin_reg.py             # Ridge Regression (HPO via Optuna)
│   │   ├── log_reg.py             # Logistic Regression (HPO via Optuna)
│   │   ├── xgboost.py             # XGBoost (GPU-accelerated, early stopping)
│   │   ├── catboost.py            # CatBoost (GPU, native categoricals, HPO)
│   │   └── tabpfn.py              # TabPFN v2 (in-context learning, GPU)
│   └── utils/
│       ├── generate_project_overview.py  # Regenerate Project Overview.pdf
│       └── generate_raw_data_dict.py     # Regenerate Raw Data Dictionary.pdf
│
├── images/DAGs/                 # Per-player causal DAG visualizations (gitignored)
├── results/                     # Model outputs (gitignored)
├── Project Overview.pdf         # Auto-generated problem statement (project root)
├── CLAUDE.md                    # Full project documentation for AI-assisted coding
├── requirements.txt
└── .gitignore
```

---

## Experiments

### Experiment 1 — Match Intensity Prediction (`notebooks/2.1.`, `scripts/Experiment1.py`)
> **Status: Reference / exploratory only**

Predicts `Match Intensity Yesterday` (continuous match performance score) from morning covariates. Retained for documentation — too many unobserved confounders (tactical, psychological, opponent quality) between training load and match outcome make this an unreliable primary causal target.

### Experiment 2 — Treatment Policy Modelling (`notebooks/2.2.`, `scripts/Experiment2.py`)
> **Primary experiment — propensity model π(Aₜ | Lₜ)**

Models the coaching staff's implicit load-assignment policy: what training intensity does the coach prescribe given this morning's player state? High R² means load decisions are systematic and recoverable from observables; the fitted model directly serves as the propensity model for downstream IPTW/MSM and doubly-robust causal estimation.

**Models:** Lin Reg, XGBoost (GPU), CatBoost (GPU), TabPFN (GPU)
**Target:** `Training Intensity Yesterday` (continuous [0,1), `target_horizon=1`)
**Key output:** `trained_model`, `X_train`, `X_test` for SHAP + propensity scoring

### Experiment 3 — Short-term Load Response (`notebooks/2.3.`, `scripts/Experiment3.py`)
> **Primary experiment — outcome model Q(A, L)**

Predicts next-day player status deterioration (`Status Decrease`, binary) from morning state. Two modes:

| Mode | Horizon | Purpose |
|------|---------|---------|
| `prediction` | t → t+1 | Early-warning system: flag at-risk players before today's session |
| `causal_framing` | t → t | Diagnostic: does yesterday's load explain today's status, controlling for morning state? |

The `causal_framing` outcome model Q(Aₜ₋₁, Lₜ) feeds directly into G-computation. The raw coefficient on Training Intensity Yesterday is biased toward zero/negative due to confounding by indication — valid causal estimates require IPTW/G-computation.

**Models:** Log Reg, XGBoost (GPU), CatBoost (GPU)

---

## Causal Framework

| Variable | Symbol | Examples |
|----------|--------|---------|
| Player covariates | **Lₜ** | Wellness z-scores, ACWR, GPS %, Days Since/Until Match |
| Treatment | **Aₜ** | Training Intensity ∈ [0,1): tanh(mean(TD%, HSD%, Dec%, Sprints%)/100) |
| Outcome | **Y** | Status Decrease (short-term), Match Intensity (long-term) |

**Causal methods:** G-computation, MSM/IPTW, G-Estimation/SNMM, Q-learning, dWOLS
**Not appropriate:** Single-stage causal meta-learners (S/T/X/DR-Learner) — these do not handle time-varying confounding affected by prior treatment.

---

## Quick Start

```python
# Install dependencies
pip install -r requirements.txt

# Regenerate processed data
python src/data/data_preprocessing.py

# Run Experiment 2 (treatment policy) with XGBoost on GPU
from scripts.Experiment2 import run_experiment, DEFAULT_COVARIATES

results = run_experiment(
    covariates=DEFAULT_COVARIATES,
    lag=3,
    model_type='xgboost',   # uses device='cuda' by default
)
print(f"R2={results['metrics']['r2']:.4f}  RMSE={results['metrics']['rmse']:.4f}")

# Run Experiment 3 (load response) — prediction mode
from scripts.Experiment3 import run_experiment as run3, DEFAULT_COVARIATES as COV3

results3 = run3(covariates=COV3, lag=3, model_type='xgboost', mode='prediction')
print(f"AUC={results3['metrics']['roc_auc']:.4f}")
```

---

## GPU Acceleration

XGBoost, CatBoost, and TabPFN all run on GPU by default when called via the experiment scripts:

| Model | GPU parameter | Requirement |
|-------|--------------|-------------|
| XGBoost | `device='cuda'` | XGBoost >= 2.0, CUDA toolkit |
| CatBoost | `task_type='GPU'` | CatBoost >= 1.2, CUDA toolkit |
| TabPFN | `device='cuda'` + `ignore_pretraining_limits=True` | PyTorch with CUDA, tabpfn >= 2.0 |

**Important**: TabPFN requires a CUDA-enabled PyTorch build. The default `pip install torch` installs CPU-only. Install CUDA PyTorch first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
(Replace `cu128` with your CUDA driver version: `cu118`, `cu121`, `cu124`, `cu128`.)

To switch to CPU: pass `device='cpu'` / `task_type='CPU'` as `model_kwargs` overrides.

---

## Data Pipeline

```
Readiness_Data.xlsx  ──┐
Raw_Data.xlsx        ──┤  data_preprocessing.py  ──>  data/processed/RTT.xlsx
Sessions.xlsx        ──┤                               (14,359 rows × 46 cols)
Games.xlsx           ──┘
```

The processed dataset merges all four raw sources. Readiness_Data is the base; Raw_Data adds detailed GPS/HR shifted +1 day ("Yesterday" columns); Games adds match performance shifted +1 day. All preprocessing is fully reproducible from the raw files.

---

## Key Dependencies

```
torch >= 2.0 (CUDA)  pandas >= 2.0       numpy >= 1.24       scipy >= 1.10
scikit-learn >= 1.3   xgboost >= 2.0      catboost >= 1.2     optuna >= 4.0
tabpfn >= 2.0         shap >= 0.44        tqdm >= 4.0
networkx >= 3.0       matplotlib >= 3.7   seaborn >= 0.12
reportlab >= 4.0      openpyxl >= 3.1
```

See `requirements.txt` for the full list. PyTorch CUDA must be installed separately (see GPU section).

---

## References

- Chakraborty & Moodie (2013). *Statistical Methods for Dynamic Treatment Regimes*. Springer.
- Chen & Guestrin (2016). XGBoost. *KDD '16*.
- Gabbett (2016). The training-injury prevention paradox. *BJSM, 50*(5), 273–280.
- Hernan & Robins (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Murphy (2003). Optimal dynamic treatment regimes. *JRSS-B, 65*(2), 331–355.
- Robins (1986). A new approach to causal inference. *Mathematical Modelling, 7*, 1393–1512.
- Wallace & Moodie (2015). Doubly-robust DTR estimation. *Biometrics, 71*(3), 636–644.

---

*KU Leuven & OH Leuven — Last updated 2026-03-26*
