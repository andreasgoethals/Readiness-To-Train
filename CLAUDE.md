# Readiness to Train - Project Documentation

**Generated:** 2026-02-10
**Project:** Causal Modeling of Player Readiness to Train
**Partnership:** KU Leuven & OH Leuven
**Purpose:** Prescriptive analytics for optimal training intensity using Causal Machine Learning

---

## Table of Contents

1. [Core Objective](#core-objective)
2. [Temporal Semantics (CRITICAL)](#temporal-semantics-critical)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Machine Learning Methods](#machine-learning-methods)
6. [Usage Guide](#usage-guide)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Core Objective

This project develops a **prescriptive analytics system** that estimates the "Readiness to Train" for individual football players. Rather than simple prediction, the project utilizes **Causal Machine Learning** to recommend the optimal **treatment** (training intensity) for each player. The ultimate aim is to maximize performance on match days while minimizing injury risk through a personalized **"traffic light" decision-support system** (Green / Orange / Red).

### Key Research Questions

1. How can we accurately estimate the probability of readiness in a "Small N, Large T" environment (27 players, 156 days)?
2. How do we effectively handle **selection bias** (covariate shift) in observational football data — coaches only give hard training to fresh players?
3. Can a Causal ML model outperform traditional sport science rules by identifying non-linear, individualized responses to training stress?

### Causal Framework

The data loader supports flexible variable selection for different experimental setups:

- **Covariates (X)**: Selected via `predictory_columns` — typically wellness z-scores, workload history, status, medical availability, Days Since Game, Days Until Match
- **Treatment (T)**: Selected via `treatment_columns` — the specific treatment variable(s) depend on the experiment. Possible candidates include Activity Type Today (session type), GPS benchmark % variables (training intensity), or combinations thereof.
- **Outcome (Y)**: Selected via `target_variable` — typically Status Decrease (binary), but can be any column

### Target Variables

The primary prediction target is **Status Decrease**, a binary indicator:
- `1`: Player's medical status worsened (Available→Attention, Available→Injured, or Attention→Injured)
- `0`: Status stable or improved

---

## Temporal Semantics (CRITICAL)

### Within-Row Temporal Ordering

Each row represents **one player-day** (date t). Variables within a row have different temporal positions — understanding this is essential for avoiding data leakage and correctly specifying causal models.

```
TIMELINE FOR A SINGLE ROW (date t):
═══════════════════════════════════════════════════════════════════════

    ┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
    │   BEFORE day t   │    │  MORNING of day t     │    │  DURING/AFTER t   │
    │   (t-1 data)     │    │  (assessment)         │    │  (decision)       │
    ├─────────────────┤    ├──────────────────────┤    ├──────────────────┤
    │ Act Type Yest.   │    │ Wellness z-scores     │    │ Activity Type     │
    │ GPS % Yesterday  │    │ Status                │    │   Today           │
    │ ACWR Yesterday   │    │ Status Decrease       │    │                   │
    │ Comment Yest.    │    │ Days Since Game       │    │ (post-assessment) │
    │ RPE Yesterday    │    │ Days Until Match      │    │                   │
    │ Med Availability │    │ Physical/Mental State │    │                   │
    │ Club Attendance  │    │ Overall Wellbeing     │    │                   │
    └─────────────────┘    └──────────────────────┘    └──────────────────┘
         FULLY KNOWN            MORNING DATA              POST-ASSESSMENT
         at day start           measured before            session type
                                activity decisions         decided AFTER
                                                           assessment (t+1)
```

### Critical Rules

1. **Activity Type Today is a post-assessment variable** — it is determined AFTER the morning assessment and derived from the next row's Activity Type Yesterday (t+1 data). It should not be used as a predictor in standard ML since it is not available at prediction time. It can be used as a treatment variable via `treatment_columns` for causal analysis.
2. **Days Since Game is NEVER 0** — it counts days since the last *completed* OHL first-team game (Activity Type = exactly "Game"). On match day, the game hasn't happened yet when the morning assessment occurs, so it counts since the *previous* match. Minimum value is 1. "Youth training or game" and "National team training/game" are NOT counted as matches.
3. **Days Until Match CAN be 0** — on match day itself, Days Until Match = 0. Only OHL first-team matches (Activity Type = exactly "Game") are counted; youth and national team games are excluded.
4. **Yesterday's data describes t-1** — all "Yesterday" columns contain data from the day before the row's date.
5. **Wellness z-scores are individualized** — each player's z-scores are relative to their own 28-day rolling baseline, NOT the team average.

### Between-Row Temporal Structure

The data is **longitudinal time-series**: each player has a sequence of daily observations. The typical weekly structure is ~6 training days followed by a match day.

For **lagged features**, the data loader creates player-grouped shifted values (e.g., `Fatigue (z)_t-1`, `Fatigue (z)_t-2`) to capture temporal patterns. These shifts are done within each player group to prevent cross-player data leakage.

---

## Project Structure

```
Readiness-To-Train/
│
├── assets/
│   └── Project Overview.pdf          # Research overview document
│
├── data/
│   ├── raw/                          # Original data files
│   │   └── Readiness_Data.csv        # Raw player monitoring data (4,239 rows × 24 cols)
│   └── processed/                    # Preprocessed data
│       ├── Readiness_Data.csv        # Cleaned & feature-engineered data (4,239 rows × 36 cols)
│       └── Processed Data Dictionary.pdf  # Auto-generated variable documentation
│
├── notebooks/
│   └── Processed_Data_Exploration.ipynb  # Comprehensive EDA notebook
│
├── src/
│   ├── data/
│   │   ├── data_preprocessing.py     # Data cleaning & feature engineering
│   │   └── data_loader.py            # Dataset creation with lags & splits
│   │
│   └── methods/
│       ├── DAG_Creator.py            # Causal DAG builder for longitudinal match cycles
│       ├── LogReg.py                 # Logistic Regression model
│       └── XGBoost.py                # XGBoost model
│
├── results/                          # Model outputs
│
└── CLAUDE.md                         # This documentation file
```

---

## Data Pipeline

### Data Asset Overview

| Metric | Value |
|--------|-------|
| Observations | 4,239 |
| Variables (raw) | 24 |
| Variables (processed) | 36 |
| Temporal Span | 156 days (Jun-Nov 2025) |
| Unique Players | 27 |

### 1. Data Preprocessing (`data_preprocessing.py`)

The preprocessing pipeline transforms raw data into analysis-ready format.

**Input:** `data/raw/Readiness_Data.csv`
**Output:** `data/processed/Readiness_Data.csv` + PDF data dictionary

**Transformations:**
1. Player ID mapping (complex keys → sequential IDs 1-27)
2. Column renaming for clarity
3. Percentage column cleaning (string → integer)
4. Comment categorization (recovery, discomfort, stiffness, etc.)
5. Composite scores (Physical State, Mental State, Overall Wellbeing)
6. Activity Type Today (session type on day t, derived from next row's Activity Type Yesterday)
7. Days Since Game (days since last *completed* match, minimum 1, never 0)
8. Days Until Match (days until next scheduled match, 0 on match day)
9. Match Day (team-level: 1 if any player has an OHL first-team game on that date; excludes youth/national team games) and Selected (player-level: 1/0/NaN)
10. Status Decrease detection (Available→Attention/Injured, Attention→Injured)
11. ACWR danger zone flagging (any ACWR > 1.5)
12. Column reordering into temporal groups
13. Save CSV + auto-generate PDF data dictionary

**Processed Dataset Columns (in order):**

| Group | Columns | Temporal Position |
|-------|---------|-------------------|
| Identifiers | Date, Playerkey, Player ID, Position | Always known |
| Historical | Medical Availability Last 14 Days, Club Attendance Last 14 Days | Before day t |
| Yesterday (t-1) | ACWR (×4), Any ACWR Danger, Activity Type Yesterday, Comment Yesterday, Comment Category Yesterday, GPS % (×5), Perceived Exertion Yesterday | Before day t |
| Morning (t) | Status, Status Decrease, Fatigue/Readiness/Soreness (z), Physical State, Sleep Quality/Stress/Mood (z), Mental State, Overall Wellbeing, Days Since Game, Days Until Match | Covariates (X) |
| Post-assessment (t) | Activity Type Today, Match Day, Selected | Session type assigned after morning assessment (t+1 data); Match Day is team-level (1 if any player has OHL first-team game — excludes youth/national team); Selected is player-level (1/0/NaN) |

**Usage:**
```python
from src.data.data_preprocessing import preprocess_data
df = preprocess_data()  # Runs full pipeline
```

### 2. Data Loading (`data_loader.py`)

The `ReadinessDataLoader` class creates ML-ready datasets with flexible variable
selection (covariates, treatment, outcome) for both standard and causal ML.

**Key Features:**
1. **Automatic Preprocessing Trigger**: Runs preprocessing if processed data missing
2. **Flexible Variable Selection**: Any column(s) can be covariates (X), treatment (T), or outcome (Y)
3. **Configurable Treatment Horizon**: Pull treatment from same or future rows (per-column)
4. **Player-Grouped Lag Features**: Safely creates temporal features per player
5. **Date-Block Splitting**: Prevents same-day leakage across splits
6. **Target Horizon**: Can shift target forward to predict future outcomes
7. **Missing Data Strategies**: Configurable for numerical/categorical features
8. **Categorical Encoding**: One-hot or label encoding
9. **Standardization**: Optional z-score normalization (fit on train only), independent for treatment
10. **Metadata Preservation**: Tracks Date, Player ID for error analysis

**Example (standard ML — no treatment):**
```python
from src.data.data_loader import ReadinessDataLoader

loader = ReadinessDataLoader()
data = loader.create_dataset(
    target_variable='Status Decrease',
    predictory_columns=[
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
        'Physical State', 'Sleep Quality (z)', 'Stress (z)',
        'Mood (z)', 'Mental State', 'Overall Wellbeing',
        'Total Distance (ACWR) Yesterday',
        'High Speed Distance (ACWR) Yesterday',
        'Any ACWR Danger',
        'Days Since Game', 'Days Until Match',
        'Medical Availability Last 14 Days',
        'Position', 'Activity Type Yesterday'
    ],
    lag=3,
    include_previous_target=True,
    test_size=0.2,
    val_size=0.1,
    categorical_encoding='one-hot',
    missing_numeric='mean',
    missing_categorical='mode'
)
# Returns: X_train, y_train, X_val, y_val, X_test, y_test, ...
```

**Example (causal ML — with treatment):**
```python
data = loader.create_dataset(
    target_variable='Status Decrease',
    predictory_columns=[
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
        'Physical State', 'Days Since Game', 'Days Until Match',
    ],
    treatment_columns=['Activity Type Today'],
    treatment_horizon=0,    # Session type is in the same row
    target_horizon=1,       # Outcome is tomorrow's Status Decrease
    lag=3,
    categorical_encoding='label'
)
# Returns: X_train, T_train, y_train, X_val, T_val, y_val, ...
```

**Example (causal ML — full treatment with mixed horizons):**
```python
data = loader.create_dataset(
    target_variable='Status Decrease',
    predictory_columns=['Fatigue (z)', 'Readiness (z)', ...],
    treatment_columns=[
        'Activity Type Today',
        'Total Distance % Yesterday',
        'High Speed Distance % Yesterday',
    ],
    treatment_horizon={
        'Activity Type Today': 0,              # Same row
        'Total Distance % Yesterday': 1,       # Pull from next row
        'High Speed Distance % Yesterday': 1,  # Pull from next row
    },
    target_horizon=1,
    standardize_treatment=False,  # Keep GPS % on original scale
)
```

---

## Machine Learning Methods

### Method 1: Logistic Regression (`LogReg.py`)

**Best For:** Interpretability, baseline performance, feature importance analysis

**Key Parameters:**
```python
LogisticRegressionModel(
    target_variable='Status Decrease',
    predictory_columns=[...],
    lag=3,
    hpo_trials=50,                # Optuna hyperparameter optimization
    class_weight='balanced',      # Auto-balance for imbalanced data
    categorical_encoding='one-hot',
    standardize=True              # Required for linear models
)
```

### Method 2: XGBoost (`XGBoost.py`)

**Best For:** Maximizing predictive performance, handling non-linear relationships

**Key Parameters:**
```python
XGBoostModel(
    target_variable='Status Decrease',
    predictory_columns=[...],
    lag=3,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=None,        # Auto-computed for imbalance
    categorical_encoding='label', # Efficient for trees
    standardize=False             # Trees don't need standardization
)
```

Both models support `target_horizon` for predicting future outcomes:
```python
model = XGBoostModel(
    target_horizon=1,  # Predict tomorrow's Status Decrease using today's features
    ...
)
```

### Causal DAG Builder (`DAG_Creator.py`)

**Best For:** Defining the causal graph structure for longitudinal causal inference methods

The `DAGCreator` class dynamically builds directed acyclic graphs (DAGs) representing the causal structure of multi-day training cycles. It is **variable-name agnostic** — all variable names are provided by the caller, nothing is hardcoded.

**Key concepts:**
- **State variables** (`state_vars`): A unified set of player state variables used as baseline (t0) and daily covariates (t1..tN). Baseline, covariates, and post-match state all represent the same concept — the player's condition measured through summarizing variables.
- **Multi-cycle support**: Multiple consecutive match cycles with variable lengths, connected by outcome-to-baseline feedback edges. Each player may have different cycle lengths depending on their match schedule.
- **Outcome feedback**: Match-day performance (outcome) at the end of each cycle causally affects the player's state entering the next cycle.

**Single cycle:**
```python
from src.methods.DAG_Creator import DAGCreator

creator = DAGCreator()
dag = creator.build_dag(
    cycle_lengths=5,                   # Single 5-day cycle
    state_vars=[                       # Player state (baseline + covariates)
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
    ],
    daily_treatment_var='Activity Type Today',  # Treatment per day
    outcome_var='Status Decrease',     # Match-day performance
)
# Returns: networkx.DiGraph with time-indexed nodes and causal edges
```

**Multi-cycle with variable lengths:**
```python
dag = creator.build_dag(
    cycle_lengths=[5, 7, 5],          # 3 cycles of different lengths
    state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
    daily_treatment_var='Activity Type Today',
    outcome_var='Status Decrease',
)
# Outcome of cycle 1 feeds back into baseline of cycle 2, etc.
feedback = creator.get_feedback_edges()
# [('Status Decrease_c1', 'Fatigue (z)_c2_t0'), ...]
```

**Node naming:** `{variable}_c{cycle}_t{day}` for state/treatment nodes, `{outcome}_c{cycle}` for outcome nodes.

**Causal edges encoded:**
| Edge type | Example | Meaning |
|-----------|---------|---------|
| `baseline_to_covariate` | Fatigue (z)_c1_t0 → Readiness (z)_c1_t1 | Initial conditions |
| `confounding` | Fatigue (z)_c1_t1 → Activity Type Today_c1_t1 | Selection bias |
| `treatment_effect` | Activity Type Today_c1_t1 → Fatigue (z)_c1_t2 | Causal effect |
| `state_carryover` | Fatigue (z)_c1_t1 → Fatigue (z)_c1_t2 | State persistence |
| `treatment_to_outcome` | Activity Type Today_c1_t5 → Status Decrease_c1 | Final effect |
| `covariate_to_outcome` | Fatigue (z)_c1_t5 → Status Decrease_c1 | Final state effect |
| `outcome_to_baseline` | Status Decrease_c1 → Fatigue (z)_c2_t0 | Inter-cycle feedback |

**Query methods:** `get_nodes_by_role()`, `get_nodes_at_time()`, `get_nodes_in_cycle()`, `get_nodes_at_cycle_day()`, `get_cycle_outcome()`, `get_feedback_edges()`, `get_parents()`, `get_children()`, `get_edges_by_relation()`, `summary()`

**Visualization methods:**

The DAGCreator supports three visualization modes for producing publication-ready figures:

| Mode | Method | Description |
|------|--------|-------------|
| `'schematic'` | `visualize(mode='schematic')` | Collapsed view: all state variables summarized into single "Player State" nodes. Clean, publication-ready. |
| `'detailed'` | `visualize(mode='detailed')` | Full expanded view: every individual variable node and all edges visible. |
| `'single_cycle'` | `visualize(mode='single_cycle', cycle=k)` | Detailed view of one specific cycle. |

Convenience shorthand: `visualize_cycle(cycle=k)` is equivalent to `visualize(mode='schematic', cycle=k)`.

**Visual encoding:**
- **Green ellipse**: Baseline state (t₀)
- **Blue ellipse**: Daily player state (covariates)
- **Orange rectangle**: Treatment (training intensity)
- **Red diamond**: Match outcome
- **Purple arrow**: Inter-cycle feedback (outcome → next baseline)
- **Dashed orange arrow**: Confounding (state → treatment, selection bias)
- **Red arrow**: Treatment effect
- **Blue arrow**: State carry-over

**Schematic visualization (single cycle):**
```python
creator = DAGCreator()
creator.build_dag(
    cycle_lengths=5,
    state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
    daily_treatment_var='Activity Type Today',
    outcome_var='Status Decrease',
)
creator.visualize(mode='schematic', save_path='results/dag_schematic.png')
```

**Schematic visualization (multi-cycle with feedback):**
```python
creator.build_dag(
    cycle_lengths=[5, 7, 5],
    state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)'],
    daily_treatment_var='Activity Type Today',
    outcome_var='Status Decrease',
)
creator.visualize(mode='schematic', save_path='results/dag_multi_cycle.png')
```

**Visualize just one cycle from a multi-cycle DAG:**
```python
creator.visualize_cycle(cycle=2, save_path='results/dag_cycle_2.png')
# or equivalently:
creator.visualize(mode='schematic', cycle=2)
```

**Detailed view (all individual variable nodes):**
```python
creator.visualize(mode='detailed', save_path='results/dag_detailed.png')
```

**Full parameter list for `visualize()`:**
```python
creator.visualize(
    mode='schematic',        # 'schematic', 'detailed', or 'single_cycle'
    cycle=None,              # Which cycle to show (None = all)
    figsize=None,            # (width, height) in inches, auto-computed if None
    save_path=None,          # Save to file (.png, .pdf, .svg)
    dpi=150,                 # Resolution for saved images
    title=None,              # Custom title (auto-generated if None)
    show=True,               # Whether to display interactively
)
```

---

## Usage Guide

### Quick Start

```python
from src.methods.XGBoost import XGBoostModel

# IMPORTANT: Only use variables available at prediction time as predictors.
# Activity Type Today is determined AFTER the morning assessment — do not include as a predictor.
predictors = [
    'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
    'Physical State', 'Sleep Quality (z)', 'Stress (z)',
    'Mood (z)', 'Mental State', 'Overall Wellbeing',
    'Total Distance (ACWR) Yesterday',
    'High Speed Distance (ACWR) Yesterday',
    'Any ACWR Danger',
    'Days Since Game', 'Days Until Match',
    'Medical Availability Last 14 Days',
    'Club Attendance Last 14 Days',
    'Position', 'Activity Type Yesterday'
]

model = XGBoostModel(
    target_variable='Status Decrease',
    predictory_columns=predictors,
    lag=3,
    include_previous_target=True,
    test_size=0.2,
    val_size=0.1
)

results = model.train()
print(f"Test ROC AUC: {results['metrics']['roc_auc']:.4f}")
```

### Running from Command Line

```bash
cd "path/to/Readiness-To-Train"
python src/data/data_preprocessing.py   # Run preprocessing
python src/methods/LogReg.py            # Run Logistic Regression
python src/methods/XGBoost.py           # Run XGBoost
python src/methods/DAG_Creator.py       # Run DAG demos + generate visualizations
```

---

## API Reference

### ReadinessDataLoader

**Constructor:**
```python
ReadinessDataLoader(data_path=None)
```

**Key Method:**
```python
create_dataset(
    target_variable: str,
    predictory_columns: List[str],
    lag: int = 0,
    include_previous_target: bool = False,
    target_horizon: int = 0,
    test_size: float = 0.2,
    val_size: float = 0.1,
    categorical_encoding: str = 'one-hot',
    missing_numeric: str = 'mean',
    missing_categorical: str = 'mode',
    standardize: bool = False,
    # Treatment variable support
    treatment_columns: Optional[List[str]] = None,
    treatment_horizon: Union[int, Dict[str, int]] = 0,
    treatment_lag: int = 0,
    standardize_treatment: Optional[bool] = None,
) -> Dict
```

**Returns (always present):**
```python
{
    'X_train': pd.DataFrame,
    'y_train': pd.Series,
    'meta_train': pd.DataFrame,   # Date, Player ID, Playerkey
    'X_val': pd.DataFrame,
    'y_val': pd.Series,
    'meta_val': pd.DataFrame,
    'X_test': pd.DataFrame,
    'y_test': pd.Series,
    'meta_test': pd.DataFrame,
    'feature_names': List[str],
    'categorical_features': List[str],
    'numerical_features': List[str],
    'encoding_info': Dict,
    'imputation_info': Dict,
    'metadata': Dict
}
```

**Returns (additional keys when treatment_columns is provided):**
```python
{
    'T_train': pd.DataFrame,              # Treatment variables for training
    'T_val': pd.DataFrame,                # Treatment variables for validation
    'T_test': pd.DataFrame,               # Treatment variables for test
    'treatment_feature_names': List[str],  # After encoding
    'treatment_categorical': List[str],
    'treatment_numerical': List[str],
    'treatment_encoding_info': Dict,
    'treatment_imputation_info': Dict,
}
```

### Model Classes

Both `LogisticRegressionModel` and `XGBoostModel` share:
```python
model.train() -> Dict  # Returns predictions, metrics, model weights
```

---

## Best Practices

### 1. Feature Selection — Temporal Safety

**Not available at prediction time (should not be used as predictors):**
- `Activity Type Today` (determined after morning assessment — t+1 data)
- Any variable derived from today's training session

**Available at prediction time (safe as predictors):**
- All wellness z-scores (morning assessment)
- All "Yesterday" columns (fully observed)
- Days Since Game, Days Until Match (known in the morning)
- Medical Availability, Club Attendance (historical)
- Position (static)

### 2. Lag Selection

- `lag=1`: Captures yesterday's influence
- `lag=3`: Short-term trends (recommended starting point)
- `lag=7`: Weekly patterns (but many more features and NaN rows)

### 3. Class Imbalance

Status decreases are rare (~3-5% of observations). Both models handle this:
- LogReg: `class_weight='balanced'`
- XGBoost: `scale_pos_weight` auto-computed from data

### 4. Treatment Variable Configuration

When using causal ML methods, use `treatment_columns` to explicitly separate treatment from covariates. The specific treatment variable(s) depend on the experiment — the data loader does not prescribe what should be the treatment. Example configurations:

| Scenario | treatment_columns | treatment_horizon | target_horizon | Research question |
|----------|------------------|-------------------|----------------|-------------------|
| Standard ML | *not set* | - | 0 | Yesterday's load → today's status |
| Future prediction | *not set* | - | 1 | Today's data → tomorrow's status |
| Session type | `['Activity Type Today']` | 0 | 1 | Effect of session type on tomorrow |
| Full prescription | `['Activity Type Today', GPS % cols]` | `{...: 0, ...: 1}` | 1 | Effect of full prescription on tomorrow |
| Yesterday's intensity | `[GPS % Yesterday cols]` | 0 | 0 | Effect of yesterday's intensity on today |

### 5. Covariate Shift Awareness

Coaches assign training based on player state (healthy players get harder training). This creates **selection bias** that makes observational data unreliable for simple prediction. The causal ML approach (SurvITE, etc.) addresses this by balancing representations across treatment groups.

---

## Troubleshooting

### Common Issues

**1. Days Since Game = 0**
This should never happen after the preprocessing fix. If you see it, re-run `python src/data/data_preprocessing.py` to regenerate the processed data.

**2. Activity Type Today used as predictor**
The data loader will emit a warning. Remove it from `predictory_columns`. This variable is determined after the morning assessment and is not available at prediction time. Use `treatment_columns` if you need it for causal analysis.

**3. All predictions are the same class**
Check class distribution and ensure `class_weight='balanced'` or `scale_pos_weight` is set.

**4. Processed data not found**
Normal — preprocessing runs automatically. Or run manually:
```bash
python src/data/data_preprocessing.py
```

---

## Theoretical Framework

- **Supercompensation**: Training stress → fatigue → recovery → adaptation cycle
- **Gabbett's U-Shaped Risk Model**: ACWR sweet spot (0.8–1.3) minimizes injury risk; >1.5 = danger zone
- **Individual Profiling**: Z-scores normalized per player (28-day rolling window)
- **Sequential Treatment Effects**: How a 5-day training sequence affects match-day readiness

## References

- Gabbett, T. J. (2016). The training-injury prevention paradox. *BJSM, 50*(5), 273-280.
- Curth et al. (2021). SurvITE: Individualized Treatment Effect estimator for Survival analysis. *NeurIPS*.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.

---

**Last Updated:** 2026-02-18
**Python Version:** 3.8+
**Key Dependencies:** pandas, numpy, scikit-learn, xgboost, optuna, matplotlib, seaborn, reportlab, networkx
