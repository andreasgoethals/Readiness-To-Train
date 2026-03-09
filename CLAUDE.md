# Readiness to Train - Project Documentation

**Generated:** 2026-02-26
**Project:** Causal Modelling of Player Readiness to Train
**Partnership:** KU Leuven & OH Leuven
**Purpose:** Prescriptive analytics for optimal training intensity using Causal Machine Learning (Dynamic Treatment Regimes)

---

## Table of Contents

1. [Core Objective](#core-objective)
2. [Temporal Semantics (CRITICAL)](#temporal-semantics-critical)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Machine Learning Methods](#machine-learning-methods)
6. [Methodological Framework](#methodological-framework)
7. [Usage Guide](#usage-guide)
8. [API Reference](#api-reference)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Core Objective

This project develops a **prescriptive analytics system** that estimates individual player **Readiness to Train** in professional football. Using **Causal Machine Learning** within a **Dynamic Treatment Regime (DTR)** framework, the system recommends an optimal daily training intensity for each player, delivered as a **continuous score between 0 and 1**. A score of 0 indicates the player should rest entirely; a score of 1 indicates the player can train at maximal intensity. This score simultaneously serves two roles: it is the recommended treatment (the prescribed training intensity from the causal model) and an observable variable for coaching and medical staff.

> **Important:** The objective is **not** injury minimisation. A policy that minimises injury risk trivially prescribes zero load. The true objective is **performance optimisation under biological constraints**, where some non-zero injury incidence may be optimal because competitive performance requires physiological stress. Injuries are part of the performance-exposure frontier, not a quantity to be driven to zero.

The system is fully **individualised**: each player receives a personalised score derived from their own physiological profile, training history, and evolving state. Because training sessions occur sequentially before each match and each player's history differs, the problem is inherently a **Dynamic Treatment Regime with variable-length decision horizons**.

### Why This Is a Causal Problem

Standard predictive modelling asks *"given the data, what will happen?"*. This project asks *"given a player's full history up to today, what training intensity today best contributes to future match-day performance?"* This requires causal reasoning for three reasons:

1. **Time-Varying Confounding Affected by Prior Treatment**: Covariates at time t (fatigue, readiness, ACWR) are simultaneously consequences of treatment at t-1 and causes of treatment at t. Classical regression cannot handle this correctly — this is precisely the setting for which G-methods were developed.
2. **Treatment-Confounder Feedback**: Coaching decisions are observational. Hard sessions are prescribed when players appear fresh; recovery is prescribed when fatigue is high. The observed training-performance association conflates physiological response with coach selection behaviour.
3. **Sequential Decision-Making**: The treatment is a repeated, time-varying intervention. The question is not just "what training today?" but "what sequence of training intensities over the cycle maximises match-day performance?" — a DTR problem.

### Key Research Questions

1. How can we accurately estimate the effect of training intensity sequences in a "Small N, Large T" environment (27 players, 156 days)?
2. How do we handle **time-varying confounding affected by prior treatment** in observational football data?
3. Can a causal DTR model identify the optimal individualised training sequence that maximises match-day performance?

### Causal Framework Variables

| Symbol | Role | Variables |
|--------|------|-----------|
| **Lₜ** (Covariates) | Player state at time t | Wellness z-scores, ACWR, composite scores, Days Since Game, Days Until Match, medical availability |
| **Aₜ** (Treatment) | Daily training intensity | Continuous score ∈ [0,1], derived from GPS metrics (total distance, high-speed distance, decelerations, sprints) normalised against individual match benchmarks |
| **Y** (Outcome) | Match-day performance | Physical intensity per minute played (continuous, already in dataset) |

The data loader supports flexible variable selection:
- **Covariates (X)**: Selected via `predictory_columns`
- **Treatment (T)**: Selected via `treatment_columns` — the specific treatment variable(s) depend on the experiment
- **Outcome (Y)**: Selected via `target_variable` — typically the match-day performance variable or Status Decrease (for binary ML experiments)

### Target Variables

For **standard ML experiments** (prediction):
- **Status Decrease** (binary): Player's medical status worsened (Available→Attention, Available→Injured, or Attention→Injured). `1` = worsened, `0` = stable/improved.

For **causal DTR experiments** (optimisation):
- **Match-day physical intensity per minute played** (continuous): The true causal outcome. The DTR objective is to find the sequence of training intensities that maximises the expected value of this outcome.

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
    │ Training Intens. │    │ Status Decrease       │    │ Selected          │
    │ ACWR Yesterday   │    │ Days Since Game       │    │                   │
    │ Comment Yest.    │    │ Days Until Match      │    │ (post-assessment) │
    │ RPE Yesterday    │    │ Match Day (schedule)  │    │                   │
    │ Med Availability │    │ Physical/Mental State │    │                   │
    │ Club Attendance  │    │ Overall Wellbeing     │    │                   │
    │ Raw GPS/HR Yest. │    │                       │    │                   │
    │ Match Perf Yest. │    │                       │    │                   │
    └─────────────────┘    └──────────────────────┘    └──────────────────┘
         FULLY KNOWN            MORNING DATA              POST-ASSESSMENT
         at day start           measured before            session decided AFTER
                                activity decisions         assessment (t+1 data)
```

### Critical Rules

1. **Activity Type Today / Training Intensity is a post-assessment variable** — it is determined AFTER the morning assessment and derived from the next row's Activity Type Yesterday (t+1 data). It should not be used as a predictor in standard ML since it is not available at prediction time. It must be used via `treatment_columns` for causal analysis.
2. **Days Since Game is NEVER 0** — it counts days since the last *completed* OHL first-team game (Activity Type = exactly "Game"). On match day, the game hasn't happened yet when the morning assessment occurs, so it counts since the *previous* match. Minimum value is 1. "Youth training or game" and "National team training/game" are NOT counted as matches.
3. **Days Until Match CAN be 0** — on match day itself, Days Until Match = 0 **for selected players only**. Unselected players' DUM points to the next match they play (player-level semantics). Only OHL first-team matches are counted. Internally, `Days Until Match` is computed using `Activity Type Today` (a post-assessment variable), but this is safe because preprocessing uses the full dataset after the fact; at inference time, Days Until Match is computed separately from the match schedule (which is known in advance).
4. **Match Day is schedule information, not post-assessment** — although derived from Activity Type Today in preprocessing, Match Day represents the team's match schedule (known in advance from the fixture list). It is classified as a morning (t) variable and is safe to use as a predictor. Only `Activity Type Today` and `Selected` are truly post-assessment variables.
5. **Yesterday's data describes t-1** — all "Yesterday" columns contain data from the day before the row's date.
6. **Wellness z-scores are individualized** — each player's z-scores are relative to their own 28-day rolling baseline, NOT the team average.

### Between-Row Temporal Structure

The data is **longitudinal time-series**: each player has a sequence of daily observations. The typical weekly structure is ~5-6 training days followed by a match day. This creates the **match cycle** structure central to the DTR formulation.

For **lagged features**, the data loader creates player-grouped shifted values (e.g., `Fatigue (z)_t-1`) to capture temporal patterns. Shifts are done within each player group to prevent cross-player data leakage.

---

## Project Structure

```
Readiness-To-Train/
│
├── Project Overview.pdf              # Research overview document (root)
├── CLAUDE.md                         # This documentation file
├── README.md                         # Repository readme
├── LICENSE                           # License file
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/
│   ├── raw/                          # Original data files (all xlsx)
│   │   ├── Readiness_Data1.xlsx      # Readiness data season 1 (4,239 rows × 24 cols, 27 players)
│   │   ├── Readiness_Data2.xlsx      # Readiness data extended (6,781 rows × 24 cols, 27 players)
│   │   ├── Raw_Data.xlsx             # Full GPS/HR/wellness data (9,968 rows × 38 cols, 84 players)
│   │   ├── Sessions.xlsx             # Session metadata (1,206 rows × 8 cols)
│   │   ├── Games.xlsx                # Match performance data (403 rows × 8 cols, 25 players)
│   │   ├── Raw Data Dictionary.pdf   # Auto-generated raw data documentation
│   │   └── NDA.pdf                   # Non-disclosure agreement
│   └── processed/                    # Preprocessed data (auto-generated)
│       ├── RTT.xlsx                  # Multi-dataset merged & feature-engineered (4,239 rows × 45 cols)
│       └── RTT Data Dictionary.pdf   # Auto-generated variable documentation
│
├── images/                           # Generated visualizations
│   └── DAGs/                         # Causal DAG visualizations (per player)
│       ├── player 1/                 # Player 1's DAGs
│       │   ├── player_1_cycle_1_schematic.png
│       │   ├── player_1_cycle_1_detailed.png
│       │   ├── player_1_all_cycles_schematic.png
│       │   └── player_1_all_cycles_detailed.png
│       └── player 2/ ... player 27/  # Players 2-27
│
├── notebooks/
│   ├── Experiment1.ipynb             # Model comparison experiments
│   └── raw_data_visualisation.ipynb  # Comprehensive raw data EDA (all 5 datasets)
│
├── scripts/
│   ├── Experiment1.py                # Configurable multi-model experiment runner
│   └── Generate_Visualizations.py    # Batch DAG generation for all players
│
├── src/
│   ├── data/
│   │   ├── data_preprocessing.py     # Multi-dataset merge & feature engineering
│   │   └── data_loader.py            # Dataset creation with lags & splits
│   │
│   └── methods/
│       ├── DAG_Creator.py            # Causal DAG builder for longitudinal match cycles
│       ├── LogReg.py                 # Logistic Regression model
│       ├── XGBoost.py                # XGBoost model
│       └── CatBoost.py              # CatBoost model with native categorical handling
│
└── results/                          # Model outputs and saved figures
```

---

## Data Pipeline

### Data Asset Overview

The project draws on **5 raw Excel datasets** from OH Leuven's player monitoring system. The preprocessing pipeline currently operates on Readiness_Data1 only; the remaining datasets are available for exploratory analysis and future integration.

| Dataset | Rows | Columns | Players | Date Range | Granularity |
|---------|------|---------|---------|------------|-------------|
| **Readiness_Data1.xlsx** | 4,239 | 24 | 27 | 2025-06-24 → 2025-11-27 (156 days) | Daily (player-day) |
| **Readiness_Data2.xlsx** | 6,781 | 24 | 27 | 2024-07-02 → 2026-02-17 (596 days) | Daily (player-day) |
| **Raw_Data.xlsx** | 9,968 | 38 | 84 | 2024-05-02 → 2026-03-01 | Session-level (player-session) |
| **Sessions.xlsx** | 1,206 | 8 | — | 2024-05-02 → 2026-03-01 | Session-level (team-session) |
| **Games.xlsx** | 403 | 8 | 25 | 2025-07-27 → 2026-02-28 | Match-level (player-match) |

**Processed dataset** (auto-generated from all datasets): 4,239 rows × 45 columns → `data/processed/RTT.xlsx`

#### Player Overlap Across Datasets

- Readiness_Data1 ∩ Readiness_Data2: **26** shared players (1 unique to each)
- Readiness_Data1 ∩ Raw_Data: **27** (all Readiness_Data1 players appear in Raw_Data)
- Raw_Data has 84 total players (57 not in the readiness datasets)
- Games has 25 players, **23** overlap with Readiness_Data1
- All 4 player-level datasets overlap: **23** players

### Raw Dataset Schemas

#### Readiness_Data1.xlsx / Readiness_Data2.xlsx (identical columns)

Both share the same 24-column structure — Readiness_Data2 is the extended temporal version.

| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Observation date |
| Playerkey | string | Hashed player identifier |
| POS | string | Playing position |
| MA% | string (%) | Medical availability last 14 days |
| Att% | string (%) | Club attendance last 14 days |
| TD, HSD, Dec >3ms², Sprints | float | ACWR (7:42 day EMA ratio) for GPS metrics |
| Reason | string | Activity reason (Training, Game, Recovery, etc.) |
| Comment | string | Free-text coaching/medical note |
| TD%, HSD%, Dec >3ms²%, Sprints%, Max Velocity% | float | GPS metrics as % of personal match benchmarks |
| rpe (z) | float | Perceived exertion z-score |
| Status | string | Medical status (Available / Attention / Injured / Sick / Absent) |
| Fatigue (z), Readiness (z), Soreness (z) | float | Physical wellness z-scores (28-day rolling) |
| Sleep Quality (z), Stress (z), Mood (z) | float | Mental wellness z-scores (28-day rolling) |

#### Raw_Data.xlsx (38 columns)

Detailed session-level GPS, heart rate, and subjective wellness data for 84 players.

| Column Group | Columns | Description |
|--------------|---------|-------------|
| Identifiers | Date_Value, start_date_time, playerkey, teamkey | Session date/time and player/team keys |
| Session Info | sessiontitle, drill_title, Reason, Comment, Detail | Session type and coaching notes |
| Duration | total_game_minutes, total_minutes | Minutes played / session duration |
| GPS Load | total_player_load, total_distance, high_speed_distance, distance_zone4, distance_zone5 | External load metrics |
| Speed/Accel | high_speed_runs, very_high_speed_runs, accelerations_zone4, decelerations_zone4, max_speed | High-intensity running and acceleration |
| Metabolic | high_metabolic_load_distance | Metabolic load distance |
| Subjective | RPE, stress_level, mood, hours_sleep, sleep_quality, readiness, muscle_soreness | Self-reported wellness (raw scores, not z-scored) |
| Heart Rate | avg_heartrate, heart_rate_exertion, max_heartrate, time_in_heartrate_zone1–6 | HR monitoring across 6 intensity zones |

#### Sessions.xlsx (8 columns)

Team-level session metadata (no player-level data).

| Column | Type | Description |
|--------|------|-------------|
| Date_Value | datetime | Session date |
| teamkey | string | Team identifier |
| matchday | int | 1 if match day, 0 otherwise |
| start_date_time | datetime | Session start timestamp |
| session_title | string | Session name |
| session_type | string | Type (e.g., Training, Match, Recovery) |
| workout_type | string | Workout classification |
| Reason | string | Session reason |

#### Games.xlsx (8 columns)

Match-level performance data per player.

| Column | Type | Description |
|--------|------|-------------|
| Team | string | Team name |
| date | datetime | Match date |
| match_week | int | Match week number |
| Game | string | Match description (opponent, competition) |
| High Intensity Per BIP (m) | float | High-intensity distance per Ball-In-Play minute |
| HIT Efforts per BIP | float | High-intensity efforts per Ball-In-Play minute |
| minutes_played | float | Total minutes played |
| playernames.playerkey | string | Hashed player identifier |

### Feature Categories (Processed Dataset — RTT.xlsx)

| Category | Features | Encoding | Source |
|----------|----------|----------|--------|
| External Load (GPS) | Total Distance, High-Speed Distance, Decelerations, Sprints | ACWR (7:42 EMA), % of personal benchmarks | Readiness_Data1 |
| Training Intensity | Training Intensity Yesterday | Composite: tanh(mean(TD%, HSD%, Dec%, Sprints%) / 100). Soft cap, range [0, 1) | Engineered from GPS % |
| Raw GPS/HR (Yesterday) | Total Minutes, Total Distance (m), High Speed Distance (m), Avg Heart Rate, Heart Rate Exertion | Absolute values, shifted +1 day | Raw_Data |
| Match Performance (Yesterday) | High Intensity Per BIP, HIT Efforts Per BIP, Minutes Played | Continuous, only filled day after match | Games |
| Subjective Wellbeing | Fatigue, Soreness, Sleep Quality, Stress, Mood | Individualised 28-day rolling-window Z-scores | Readiness_Data1 |
| Composite Scores | Physical State, Mental State, Overall Wellbeing | Aggregated from subjective sub-scales | Engineered |
| Contextual / Medical | Medical availability %, Club attendance %, Activity reason, Medical status | Categorical | Readiness_Data1 |
| Temporal Context | Days Since Game, Days Until Match | Integer; Days Since Game ≥ 1, Days Until Match ≥ 0 |

### 1. Data Preprocessing (`data_preprocessing.py`)

The preprocessing pipeline merges all raw datasets into a single analysis-ready file.

**Input:** `data/raw/Readiness_Data1.xlsx`, `Raw_Data.xlsx`, `Sessions.xlsx`, `Games.xlsx`
**Output:** `data/processed/RTT.xlsx` + `RTT Data Dictionary.pdf`

**Transformations:**
1. Load all raw xlsx files (Readiness_Data1 as base, Raw_Data, Sessions, Games)
2. Player ID mapping (complex keys → sequential IDs 1-27)
3. Column renaming for clarity
4. Percentage column cleaning (string → integer)
5. Comment categorization (recovery, discomfort, stiffness, etc.)
6. **Merge Raw_Data** GPS/HR columns (total_minutes, total_distance, high_speed_distance, avg_heartrate, heart_rate_exertion) — aggregated per player-day (sum for volume, weighted mean for HR), shifted +1 day so day-of data becomes "yesterday"
7. **Merge Games** match performance columns (High Intensity Per BIP, HIT Efforts Per BIP, minutes_played) — shifted +1 day (match data → day after match)
8. Composite scores (Physical State, Mental State, Overall Wellbeing)
9. Activity Type Today (session type on day t, derived from next row's Activity Type Yesterday)
10. Days Since Game (days since last *completed* match, minimum 1, never 0)
11. Days Until Match (days until next scheduled match, 0 on match day)
12. Match Day (team-level) and Selected (player-level)
13. Status Decrease detection
14. ACWR danger zone flagging (any ACWR > 1.5)
15. Training Intensity Yesterday composite (tanh(mean(TD%, HSD%, Dec%, Sprints%) / 100), soft cap in [0, 1))
16. Column reordering into temporal groups
17. Save RTT.xlsx + auto-generate PDF data dictionary

**Processed Dataset Columns (45 columns, in order):**

| Group | Columns | Temporal Position | Source |
|-------|---------|-------------------|--------|
| Identifiers | Date, Playerkey, Player ID, Position | Always known | RD1 |
| Historical | Medical Availability Last 14 Days, Club Attendance Last 14 Days | Before day t | RD1 |
| Yesterday (t-1) RD1 | ACWR (×4), Any ACWR Danger, Activity Type Yesterday, Comment Yesterday, Comment Category Yesterday, GPS % (×5), Training Intensity Yesterday, Perceived Exertion Yesterday | Before day t | RD1 |
| Yesterday (t-1) Raw | Total Minutes, Total Distance (m), High Speed Distance (m), Avg Heart Rate, Heart Rate Exertion | Before day t | Raw_Data (shifted) |
| Yesterday (t-1) Games | Match High Intensity Per BIP, Match HIT Efforts Per BIP, Match Minutes Played | Before day t (only day after match) | Games (shifted) |
| Morning (t) | Status, Status Decrease, Fatigue/Readiness/Soreness (z), Physical State, Sleep Quality/Stress/Mood (z), Mental State, Overall Wellbeing, Days Since Game, Days Until Match, Match Day | Covariates Lₜ (Match Day = schedule info, known in advance) | RD1 + Engineered |
| Post-assessment (t) | Activity Type Today, Selected | Treatment Aₜ — assigned after morning assessment | Engineered |

**Usage:**
```python
from src.data.data_preprocessing import preprocess_data
df = preprocess_data()  # Runs full pipeline, saves RTT.xlsx
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

### Method 3: CatBoost (`CatBoost.py`)

**Best For:** Native categorical feature handling, robust gradient boosting with automatic overfitting detection

**Key Parameters:**
```python
CatBoostModel(
    target_variable='Status Decrease',
    predictory_columns=[...],
    lag=3,
    hpo_trials=20,                # Optuna hyperparameter optimization
    iterations=500,
    depth=6,
    learning_rate=0.1,
    categorical_encoding='label', # CatBoost handles categoricals natively
    standardize=False             # Trees don't need standardization
)
```

All three models support `target_horizon` for predicting future outcomes:
```python
model = XGBoostModel(
    target_horizon=1,  # Predict tomorrow's Status Decrease using today's features
    ...
)
```

### Causal DAG Builder (`DAG_Creator.py`)

**Best For:** Defining the causal graph structure for longitudinal causal inference and DTR methods

The `DAGCreator` class is **player-specific**: instantiated once per player, it auto-loads that player's data, auto-detects match cycle boundaries, and builds the full longitudinal DAG for all cycles in the data. It is **variable-name agnostic** — all variable names are provided by the caller, nothing is hardcoded.

**Key concepts:**
- **State variables** (`state_vars`): Player state Lₜ — daily covariates (t=1..N) in each cycle. The first state (t=1) represents the player's condition at cycle start. Represent the player's condition: wellness z-scores, ACWR, composite scores, temporal context.
- **Treatment** (`treatment_var`): Daily training intensity Aₜ — a continuous score [0,1] derived from GPS metrics normalised against individual match benchmarks.
- **Outcome** (`outcome_var`): Match-day performance Y — physical intensity per minute played (continuous, to be maximised).
- **Player-specific cycles**: Cycle boundaries are detected from the player's own `Activity Type Today == 'Game'` rows. Days where the team plays but the player is NOT selected simply extend the current cycle — they are not treated as a match boundary for that player.
- **`cross_var_carryover`**: If `True`, every state variable at time t causally influences all state variables at t+1 (N² edges). If `False` (default), each variable only influences itself at t+1 (N edges). Applied consistently to all day→day+1 transitions within each cycle.

**Constructor:**
```python
from src.methods.DAG_Creator import DAGCreator

creator = DAGCreator(
    player_id=1,
    state_vars=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
                'Days Since Game', 'Days Until Match'],
    treatment_var='Training Intensity Score',
    outcome_var='Match Performance',
    cross_var_carryover=False,   # self-only carryover (default)
    data_path=None,              # uses default processed data path
)
# DAG is built automatically — creator.dag is a networkx.DiGraph
```

**Node naming:** `{variable}_c{cycle}_t{day}` for state/treatment nodes, `{outcome}_c{cycle}` for outcome nodes.

**Causal edges encoded:**

| Edge type | Meaning |
| --------- | ------- |
| `confounding` | Lₜ → Aₜ (coaches prescribe based on player state — selection bias) |
| `treatment_effect` | Aₜ → Lₜ₊₁ (training changes fatigue, soreness, adaptation) |
| `state_carryover` | Lₜ → Lₜ₊₁ (state persistence; scope controlled by `cross_var_carryover`) |
| `treatment_to_outcome` | Aₜ → Y (final session effect on match performance) |
| `covariate_to_outcome` | Lₜ → Y (final state effect on match performance) |
| `outcome_to_first_state` | Y_k → L₁_{k+1} (match toll feeds back into next cycle's first state) |
| `final_state_to_first_state` | L_final_k → L₁_{k+1} (final player state carries over to next cycle) |

**Query methods:** `get_nodes_by_role()`, `get_nodes_at_time()`, `get_nodes_in_cycle()`, `get_nodes_at_cycle_day()`, `get_cycle_outcome()`, `get_feedback_edges()`, `get_parents()`, `get_children()`, `get_edges_by_relation()`, `get_time_varying_confounders()`, `to_adjacency_matrix()`, `summary()`

**Visualization:**

```python
creator.visualize(
    cycles=None,          # None=all cycles, int=one cycle index, List[int]=subset
    completeness='schematic',  # 'schematic' or 'detailed'
    save_path='results/dag_schematic.png',  # Explicit path (takes precedence)
    output_dir='images/DAGs/player 1/',             # Directory with auto-generated filename
    dpi=150,
    show=True
)
```

If `output_dir` is provided and `save_path` is not, the filename is auto-generated as
`player_{id}_{cycle_tag}_{completeness}.png` inside the specified directory.

**`completeness` options:**

- **`'schematic'`**: Collapsed view — all state variables summarized into a single "Player State" node per time step. Clean, publication-ready.
- **`'detailed'`**: Full expanded view — every individual variable shown as a separate node, enclosed by a labeled "Player State" bounding box per time step.

**Visual encoding:**
- **Blue ellipse**: Player state Lₜ (covariates)
- **Amber rectangle**: Treatment Aₜ (training intensity [0,1])
- **Crimson diamond**: Match outcome Y (performance)
- **Purple arrow**: Inter-cycle feedback (outcome → next cycle's first state)
- **Teal arrow**: Final state carry-over (last player state → next cycle's first state)
- **Dashed amber arrow**: Confounding (state → treatment, selection bias)
- **Red arrow**: Treatment effect (Aₜ → Lₜ₊₁)
- **Blue arrow**: State carry-over (Lₜ → Lₜ₊₁)

**Examples:**
```python
# Explicit save path
creator.visualize(cycles=None, completeness='schematic', save_path='results/dag_all.png')

# Auto-generated filename in output directory
creator.visualize(cycles=1, completeness='detailed', output_dir='images/DAGs/player 1/')
# -> saves as images/DAGs/player 1/player_1_cycle_1_detailed.png

# Query the DAG object
adj = creator.to_adjacency_matrix()
tvc = creator.get_time_varying_confounders()
print(creator.summary())
```

---

## Methodological Framework

### Dynamic Treatment Regimes (DTR)

A DTR is a sequence of decision rules d = (d₁, …, dK), one per decision point (training day), mapping from a player's evolving covariate and treatment history to an optimal treatment action. The goal is the **optimal DTR**: the regime that maximises expected match-day performance.

### G-Methods for Time-Varying Confounding

The causal estimation strategy uses G-methods (Robins, 1986), designed for settings with time-varying confounders affected by prior treatment:

| Method | Description | Suitability |
|--------|-------------|-------------|
| **G-Computation** | Models joint distribution of all post-treatment variables; simulates counterfactuals under hypothetical regimes | Good (parametric) |
| **MSM / IPTW** | Marginal Structural Models with Inverse Probability of Treatment Weighting; creates pseudo-population independent of confounders | Good (parametric) |
| **G-Estimation / SNMM** | Structural Nested Mean Models targeting the blip function; doubly robust | Good (semi-parametric) |

### DTR Estimation Methods

| Method | Sequential Opt. | Continuous Tx | Small N |
|--------|----------------|---------------|---------|
| **Q-Learning** | Yes (backward induction) | Yes | Moderate |
| **dWOLS** | Yes (native) | Yes (extended) | Good |
| Off-Policy RL | Yes (native) | Yes | Limited (data-hungry) |

> **Note:** Causal meta-learners (S-Learner, T-Learner, X-Learner, DR-Learner) are **not appropriate** for this problem. They estimate single-stage CATEs and do not handle time-varying confounding affected by prior treatment, nor do they optimise over sequential decisions.

### Causal Identification Assumptions

1. **Sequential exchangeability**: At each time point, treatment is independent of potential outcomes conditional on observed history. Plausible given rich covariates (GPS, wellness, medical status), but unmeasured factors (tactical, personal) may threaten this.
2. **Positivity (overlap)**: For every covariate history, there must be a positive probability of receiving any treatment level. Requires sufficient variation in prescribed intensities across player states.
3. **Consistency**: The observed outcome under the treatment actually received equals the potential outcome under that treatment.

### Statistical Regime: Small N, Large T

Only 27 players but moderately long time series (~156 days each). The signal is likely dominated by **within-player dynamics**. Models must be parsimonious or leverage partial pooling to avoid overfitting. Methods should balance individual-level tailoring with limited sample size.

---

## Usage Guide

### Quick Start

```python
from src.methods.XGBoost import XGBoostModel

# IMPORTANT: Only use variables available at prediction time as predictors.
# Activity Type Today / Training Intensity is determined AFTER the morning assessment.
predictors = [
    'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
    'Physical State', 'Sleep Quality (z)', 'Stress (z)',
    'Mood (z)', 'Mental State', 'Overall Wellbeing',
    'Total Distance (ACWR) Yesterday',
    'High Speed Distance (ACWR) Yesterday',
    'Any ACWR Danger',
    'Training Intensity Yesterday',
    'Days Since Game', 'Days Until Match',
    'Medical Availability Last 14 Days',
    'Club Attendance Last 14 Days',
    'Position', 'Activity Type Yesterday',
    # New columns from Raw_Data (GPS/HR)
    'Total Minutes Yesterday',
    'Total Distance (m) Yesterday',
    'High Speed Distance (m) Yesterday',
    'Avg Heart Rate Yesterday',
    'Heart Rate Exertion Yesterday',
    # New columns from Games (match performance — only filled day after match)
    'Match High Intensity Per BIP Yesterday',
    'Match HIT Efforts Per BIP Yesterday',
    'Match Minutes Played Yesterday',
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

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `Experiment1.ipynb` | Interactive model comparison experiments (LogReg, XGBoost, CatBoost) |
| `raw_data_visualisation.ipynb` | Comprehensive EDA across all 5 raw datasets: missingness analysis, dataset linkage (Venn diagrams, ER diagram), temporal coverage, wellness/GPS distributions, player profiles (radar charts), and cross-dataset correlation analysis |

### Running from Command Line

```bash
cd "path/to/Readiness-To-Train"
python src/data/data_preprocessing.py         # Run multi-dataset preprocessing -> RTT.xlsx
python src/methods/LogReg.py                  # Run Logistic Regression
python src/methods/XGBoost.py                 # Run XGBoost
python src/methods/DAG_Creator.py             # Run DAG demos + generate visualizations
python scripts/Experiment1.py                 # Run all-model comparison experiment
python scripts/Generate_Visualizations.py     # Generate DAGs for all 27 players
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
    'T_train': pd.DataFrame,
    'T_val': pd.DataFrame,
    'T_test': pd.DataFrame,
    'treatment_feature_names': List[str],
    'treatment_categorical': List[str],
    'treatment_numerical': List[str],
    'treatment_encoding_info': Dict,
    'treatment_imputation_info': Dict,
}
```

### Model Classes

`LogisticRegressionModel`, `XGBoostModel`, and `CatBoostModel` all share:
```python
model.train() -> Dict  # Returns predictions, metrics, model weights
```

### Experiment Runner (`scripts/Experiment1.py`)

```python
from scripts.Experiment1 import run_experiment

# Inclusion-based: specify exact predictors
results = run_experiment(
    target='Status Decrease',
    lag=3,
    predictory_columns=['Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
                        'Physical State', 'Days Since Game'],
)

# Exclusion-based: exclude specific columns, use all others
results = run_experiment(
    target='Status Decrease',
    lag=3,
    exclude_columns=['Comment Yesterday', 'Activity Type Today'],
)
```

**Parameters:** `target`, `lag`, `predictory_columns`, `exclude_columns`, `target_horizon`, `test_size`, `val_size`, `hpo_trials`, `random_state`, `verbose`.

**Returns:** `{'config': {...}, 'models': {...}, 'comparison': [...], 'summary': {...}}`

### DAG Batch Generator (`scripts/Generate_Visualizations.py`)

```python
from scripts.Generate_Visualizations import generate_all_player_dags
generate_all_player_dags()  # Generates 4 DAGs per player into images/DAGs/player <id>/
```

---

## Best Practices

### 1. Feature Selection — Temporal Safety

**Not available at prediction time (must not be used as predictors):**
- `Activity Type Today` (determined after morning assessment — t+1 data)
- `Selected` (coach's squad decision — post-assessment)
- Any GPS metric from today's training session
- Training Intensity Score (post-assessment)

**Available at prediction time (safe as predictors / covariates Lₜ):**
- All wellness z-scores (morning assessment)
- All "Yesterday" columns from Readiness_Data1 (ACWR, GPS %, Training Intensity Yesterday, RPE, comments)
- All "Yesterday" columns from Raw_Data (Total Minutes, Total Distance (m), High Speed Distance (m), Avg Heart Rate, Heart Rate Exertion)
- All "Yesterday" columns from Games (Match High Intensity Per BIP, Match HIT Efforts Per BIP, Match Minutes Played) — only filled day after match
- Days Since Game, Days Until Match, Match Day (known in the morning from schedule)
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

| Scenario | treatment_columns | treatment_horizon | target_horizon | Research question |
|----------|------------------|-------------------|----------------|-------------------|
| Standard ML | *not set* | - | 0 | Yesterday's load → today's status |
| Future prediction | *not set* | - | 1 | Today's data → tomorrow's status |
| Session type | `['Activity Type Today']` | 0 | 1 | Effect of session type on tomorrow |
| Training intensity | `['Training Intensity Yesterday']` | 1 | 1 | Effect of today's intensity on tomorrow |
| Full prescription | `['Activity Type Today', 'Training Intensity Yesterday']` | `{...: 0, ...: 1}` | 1 | Effect of full prescription on tomorrow |
| Yesterday's intensity | `['Training Intensity Yesterday']` | 0 | 0 | Effect of yesterday's intensity on today |

### 5. Covariate Shift and Confounding

Coaches assign training based on player state (healthy players get harder training). This creates **time-varying confounding affected by prior treatment** — the core methodological challenge. Standard regression and single-stage causal estimators (meta-learners) cannot handle this correctly. The appropriate methods are G-computation, MSMs with IPTW, G-estimation, Q-learning, or dWOLS — all designed for this specific type of confounding.

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

- **Supercompensation**: Training stress → fatigue → recovery → adaptation. ACWR captures where a player sits on the stress-recovery-adaptation curve. Optimal intensity maximises the supercompensation response — neither zero nor maximal.
- **Gabbett's U-Shaped Risk Model**: ACWR sweet spot (0.8–1.3) maximises fitness gains without spiking injury risk; ACWR > 1.5 = danger zone.
- **Individual Profiling**: Z-scores normalized per player (28-day rolling window); GPS metrics as % of personal match benchmarks.
- **Sequential Treatment Effects**: A single session does not determine match readiness — it is the sequence across the cycle that matters. This is precisely what the DTR framework captures.
- **Dynamic Treatment Regime**: The optimal decision at time t depends on the full history of states and treatments up to t. The DTR maps this history to an optimal training intensity at each decision point.

## References

- Chakraborty, B. & Moodie, E. E. M. (2013). *Statistical Methods for Dynamic Treatment Regimes*. Springer.
- Gabbett, T. J. (2016). The training-injury prevention paradox. *BJSM, 50*(5), 273-280.
- Hernan, M. A. & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Hernan, M. A., Brumback, B., & Robins, J. M. (2001). Marginal structural models to estimate the joint causal effect of nonrandomized treatments. *JASA, 96*(454), 440-448.
- Murphy, S. A. (2003). Optimal dynamic treatment regimes. *JRSS-B, 65*(2), 331-355.
- Robins, J. M. (1986). A new approach to causal inference in mortality studies with sustained exposure periods. *Mathematical Modelling, 7*(9-12), 1393-1512.
- Robins, J. M., Hernan, M. A., & Brumback, B. (2000). Marginal structural models and causal inference in epidemiology. *Epidemiology, 11*(5), 550-560.
- Simoneau, G., Moodie, E. E. M., Nijjar, J. S., & Platt, R. W. (2020). Estimating optimal dynamic treatment regimes with survival outcomes. *JASA, 115*(531), 1531-1539.
- Wallace, M. P. & Moodie, E. E. M. (2015). Doubly-robust dynamic treatment regimen estimation via weighted least squares. *Biometrics, 71*(3), 636-644.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.

---

**Last Updated:** 2026-03-07
**Python Version:** 3.8+
**Key Dependencies:** pandas, numpy, scipy, scikit-learn, xgboost, catboost, optuna, matplotlib, seaborn, missingno, matplotlib-venn, plotly, reportlab, openpyxl, networkx, jinja2
