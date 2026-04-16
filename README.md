# Readiness to Train

### Causal Modelling of Player Readiness to Train in Professional Football

A collaboration between **KU Leuven** and **OH Leuven** exploring whether causal machine learning can optimise daily training intensity decisions for professional football players.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)

---

## The Problem

In professional football, coaching staff make daily decisions about how hard each player should train. These decisions balance competing objectives: train too little and the player arrives at match day underprepared; train too hard and the player arrives fatigued or injured. The optimal training intensity depends on each player's current physiological state, their recent load history, and their position in the match cycle.

This project asks: **can we build a data-driven system that recommends the optimal training intensity for each player, each day, to maximise match-day performance?**

This is fundamentally a **causal** question. Coaches already adjust training based on player state (fresh players get harder sessions), which means naive correlations between training load and performance are confounded by the coach's selection behaviour. Disentangling the true effect of training from the coach's judgment requires causal inference methods.

---

## The Approach

We framed the problem as a **Dynamic Treatment Regime (DTR)** --- a sequential decision-making framework from causal inference where the "treatment" (training intensity) is prescribed daily based on evolving patient/player state. The project proceeded in three stages:

1. **Can we predict match performance?** (Experiment 1) --- First, we tested whether match-day performance is even predictable from training data. If not, it cannot serve as a causal optimisation target.

2. **Can we predict coaching decisions?** (Experiment 2) --- After match performance proved unpredictable, we pivoted: instead of optimising the outcome directly, we modelled the coaching staff's implicit load-assignment policy. If the model accurately predicts what an experienced coach would prescribe, it effectively encodes their expertise as a continuous *Readiness to Train* score.

3. **Can we predict player health transitions?** (Experiment 3) --- As an auxiliary experiment, we investigated whether next-day player status deterioration is predictable from morning state.

---

## Key Results

### Experiment 1: Match Intensity is Not Predictable

The best model (Ridge Regression) achieved R2 = 0.27 on raw match intensity --- but **permutation importance revealed this was entirely driven by Player ID** (some players consistently perform higher than others). A follow-up experiment on *personal deviation* (each player vs. their own baseline) showed **R2 near zero for all models**: no within-player signal exists.

> **Implication:** Match-day performance cannot serve as a causal target. Too many unobserved factors (tactics, opponent, psychology) intervene between training and match output.

### Experiment 2: Coaching Decisions Are Recoverable (R2 = 0.43)

The best model (**TabPFN, lag=3**) explains **43% of the variance** in coaching training intensity decisions (Pearson r = 0.66). The model works consistently: **20 of 22 test-set players** have positive per-player R2.

**Top predictive features:** Days Until Match (periodisation structure), GPS load history (Total Distance %), and heart rate metrics. This confirms that coaching decisions are systematic and driven by observable physiological signals.

> **Implication:** The model serves as a data-driven *Readiness to Train* proxy, encoding the coaching staff's collective expertise into a continuous score.

### Experiment 3: Status Decrease Prediction (AUC = 0.64)

Modest discriminative ability above random, but limited by severe class imbalance (~2% positive rate, only 18 events in test set).

---

## Data

The project uses **daily monitoring data** from OH Leuven covering **28 first-team players** over ~20 months (July 2024 -- February 2026). Four raw datasets are merged into a single analysis-ready file:

| Dataset | Rows | Players | Content |
|---------|------|---------|---------|
| Readiness_Data | 14,359 | 28 | Wellness z-scores, ACWR, GPS benchmarks, medical status |
| Raw_Data | 9,968 | 84 | Detailed GPS, heart rate, session metadata |
| Sessions | 1,206 | --- | Match day flags, session types |
| Games | 403 | 24 | High-intensity distance/efforts per ball-in-play |

**Processed dataset:** `data/processed/RTT.xlsx` (5,235 useful rows, 46 engineered features). Full variable documentation in `data/raw/Raw Data Dictionary.pdf` and `data/processed/RTT Data Dictionary.pdf`.

**Key engineered variables:**
- **Training Intensity** [0, 1): `tanh(harmonic_mean(TD%, HSD%, Dec%, Sprints%) / 100)` --- the coaching staff's daily load prescription
- **Match Intensity**: geometric mean of high-intensity distance and efforts per ball-in-play minute, scaled by playing time
- **Status Decrease**: binary indicator of next-day medical status worsening (~2% prevalence)

---

## Repository Structure

```
Readiness-To-Train/
├── data/
│   ├── raw/                          # Original xlsx + Raw Data Dictionary.pdf
│   └── processed/                    # RTT.xlsx + RTT Data Dictionary.pdf
│
├── notebooks/
│   ├── 0. Processed_Data_Quality     # Data quality checks
│   ├── 0. TI_Missingness_Analysis    # Training Intensity NaN analysis
│   ├── 1.1. Match Analysis           # Match-level EDA
│   ├── 1.2. Raw Data Visualisation   # EDA across all 4 raw datasets
│   ├── 1.3. Processed Data Visualisation  # EDA of processed RTT.xlsx
│   ├── 2.1. Experiment1              # Match Intensity prediction
│   ├── 2.2. Experiment2              # Training Intensity prediction
│   └── 2.3. Experiment3              # Status Decrease prediction
│
├── scripts/                          # Experiment runner scripts
│   ├── Experiment1.py
│   ├── Experiment2.py
│   └── Experiment3.py
│
├── src/
│   ├── data/
│   │   ├── data_preprocessing.py     # Multi-dataset merge & feature engineering
│   │   └── data_loader.py            # ML-ready dataset creation (lags, splits)
│   ├── methods/
│   │   └── dag_creator.py            # Causal DAG builder (player-specific)
│   ├── models/
│   │   ├── lin_reg.py                # Ridge Regression (Optuna HPO)
│   │   ├── log_reg.py                # Logistic Regression (Optuna HPO)
│   │   ├── xgboost.py                # XGBoost (GPU, early stopping)
│   │   ├── catboost.py               # CatBoost (GPU, native categoricals)
│   │   └── tabpfn.py                 # TabPFN v2 (in-context learning, GPU)
│   └── utils/
│       ├── generate_project_overview.py
│       ├── generate_project_results.py
│       ├── generate_visualizations.py
│       └── generate_raw_data_dict.py
│
├── images/DAGs/                      # 499 causal DAG visualizations (all players, all cycles)
├── Project Overview.pdf              # Problem statement & research design
├── Project Results.pdf               # Experimental findings & conclusions
└── requirements.txt
```

---

## Models

| Model | Type | GPU | Strengths |
|-------|------|-----|-----------|
| **Ridge Regression** | Linear | No | Interpretable coefficients, fast, SHAP-compatible |
| **Logistic Regression** | Linear | No | Log-odds interpretation, balanced class weights |
| **XGBoost** | Tree ensemble | `device='cuda'` | High performance, SHAP TreeExplainer |
| **CatBoost** | Tree ensemble | `task_type='GPU'` | Native categorical handling, robust HPO |
| **TabPFN v2** | Transformer | `device='cuda'` | In-context learning, no iterative training, excels on small datasets |

---

## Quick Start

```bash
# Install PyTorch with CUDA (required for TabPFN GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt

# Regenerate processed data from raw files
python src/data/data_preprocessing.py

# Run experiments
python scripts/Experiment1.py
python scripts/Experiment2.py
python scripts/Experiment3.py

# Generate reports and DAG visualizations
python src/utils/generate_project_results.py
python src/utils/generate_project_overview.py
python src/utils/generate_visualizations.py
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

*KU Leuven & OH Leuven*
