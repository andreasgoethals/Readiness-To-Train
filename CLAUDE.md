# Readiness to Train - Project Documentation

**Generated:** 2026-02-26  **Last Updated:** 2026-04-16
**Project:** Causal Modelling of Player Readiness to Train
**Partnership:** KU Leuven & OH Leuven
**Purpose:** Causal analytics for daily training load decisions using longitudinal observational panel data

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

1. **Treatment policy modelling**: What morning-state features predict the training intensity assigned by the coaching staff? (models the propensity $\pi(A_t \mid L_t)$ — foundation for IPTW)
2. **Short-term load response**: Does today's training intensity causally predict next-day status deterioration? (G-computation / IPTW outcome model)
3. **Handling time-varying confounding**: How do we correctly estimate causal effects in the presence of time-varying confounding affected by prior treatment?
4. **Sequential optimisation**: What sequence of training intensities over a match cycle maximises within-player performance? (DTR / Q-learning / dWOLS)

> **Note on match-day performance as outcome:** Directly optimising match-day performance (e.g., Match Intensity Yesterday) as the downstream causal target proves intractable in practice. Too many unobserved confounders — tactical decisions, opponent quality, team composition, psychological factors — sit between the training process and the match result, making the causal chain from daily load to match outcome largely unidentifiable from available data. Research therefore focuses on identifiable short-to-medium-term outcomes (training intensity assignment, next-day status transitions) before attempting longer-horizon questions.

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

| Experiment | Target | Type | Use |
|---|---|---|---|
| **Exp 2 — Treatment Policy** | `Training Intensity Yesterday` (horizon=1) | Regression [0,1) | Propensity model; coaching policy description |
| **Exp 3 — Load Response** | `Status Decrease` (horizon=1) | Binary classification | Short-term outcome model; G-computation input |
| **Future DTR** | `Match Intensity Yesterday` | Regression [0,∞) | Long-horizon causal outcome (identifiability limited) |

**Status Decrease** (binary): Player's medical status worsened (Available→Attention, Available→Injured, or Attention→Injured). `1` = worsened, `0` = stable/improved. Prevalence ≈ 3–5%.

**Training Intensity Yesterday** (regression target in Exp 2): Composite score = `tanh(mean(TD%, HSD%, Dec%, Sprints%) / 100)`, soft-capped in `[0, 1)`. With `target_horizon=1`, this represents today's intensity assigned after the morning assessment.

**Match Intensity Yesterday** (reference only): `sqrt(Match HID Per BIP Yesterday × Match HIE Per BIP Yesterday) × sqrt(clip(minutes_played, 15, 90) / 90)`. Range ≈ [0, ∞), only filled on the day after a match. Retained in the dataset as a feature and for exploratory analysis, but too many unobserved confounders make it an unreliable primary causal target.

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
├── Project Overview.pdf              # Problem statement & research design (root)
├── Project Results.pdf               # Experimental findings & conclusions (root)
├── CLAUDE.md                         # This documentation file
├── README.md                         # Repository readme
├── LICENSE                           # License file
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/
│   ├── raw/                          # Original data files (all xlsx)
│   │   ├── Readiness_Data.xlsx       # Readiness monitoring data (14,359 rows × 24 cols, 28 players)
│   │   ├── Raw_Data.xlsx             # Full GPS/HR/wellness data (9,968 rows × 39 cols, 84 players)
│   │   ├── Sessions.xlsx             # Session metadata (1,206 rows × 9 cols)
│   │   ├── Games.xlsx                # Match performance data (403 rows × 9 cols, 24 players)
│   │   ├── Raw Data Dictionary.pdf   # Auto-generated raw data documentation
│   │   └── NDA.pdf                   # Non-disclosure agreement
│   └── processed/                    # Preprocessed data (auto-generated)
│       ├── RTT.xlsx                  # Multi-dataset merged & feature-engineered (14,359 rows × 46 cols)
│       └── RTT Data Dictionary.pdf   # Auto-generated variable documentation
│
├── images/                           # Generated visualizations
│   └── DAGs/                         # Causal DAG visualizations (per player)
│       ├── player 1/                 # Player 1's DAGs
│       │   ├── player_1_cycle_1_schematic.png
│       │   ├── player_1_cycle_1_detailed.png
│       │   ├── player_1_all_cycles_schematic.png
│       │   └── player_1_all_cycles_detailed.png
│       └── player 2/ ... player 28/  # Players 2-28
│
├── notebooks/
│   │
│   │   # 0.x — Debugging / temporary quality checks
│   ├── 1.1. Match Analysis.ipynb           # Match-level data exploration and analysis
│   ├── 0. Processed_Data_Quality.ipynb    # Automated quality checks on RTT.xlsx
│   ├── 0. TI_Missingness_Analysis.ipynb   # Training Intensity Yesterday missingness
│   │
│   │   # 1.x — Data visualisation (EDA)
│   ├── 1.2. Raw Data Visualisation.ipynb      # EDA across all 4 raw datasets
│   └── 1.3. Processed Data Visualisation.ipynb # EDA of processed RTT.xlsx
│
│   # 2.x — Experiments (one number per experiment)
│   ├── 2.1. Experiment1.ipynb             # Exp 1: Match Intensity prediction (reference/exploratory)
│   ├── 2.2. Experiment2.ipynb             # Exp 2: Treatment policy modelling (Training Intensity)
│   └── 2.3. Experiment3.ipynb             # Exp 3: Short-term load response (Status Decrease)
│
├── scripts/
│   ├── Experiment1.py               # Exp 1: predict Match Intensity (reference/exploratory only)
│   ├── Experiment2.py               # Exp 2: treatment policy modelling (Training Intensity prediction)
│   ├── Experiment3.py               # Exp 3: short-term load response (Status Decrease prediction)
│   └── Experiment3.py                # Experiment 3 runner (Status Decrease)
│
├── src/
│   ├── data/
│   │   ├── data_preprocessing.py    # Data cleaning & feature engineering
│   │   └── data_loader.py           # Dataset creation with lags & splits
│   │
│   ├── methods/
│   │   └── dag_creator.py           # Causal DAG builder for longitudinal match cycles
│   │
│   ├── models/                      # ML model classes (all share the same .train() interface)
│   │   ├── __init__.py
│   │   ├── lin_reg.py               # Ridge Regression (HPO via Optuna, regression + classification)
│   │   ├── log_reg.py               # Logistic Regression (HPO via Optuna)
│   │   ├── xgboost.py               # XGBoost (early stopping)
│   │   ├── catboost.py              # CatBoost (native categoricals + HPO)
│   │   └── tabpfn.py                # TabPFN v2 (in-context learning, no HPO)
│   │
│   └── utils/                       # Utility scripts for PDF generation
│       ├── generate_project_overview.py  # Generate Project Overview.pdf
│       ├── generate_project_results.py  # Generate Project Results.pdf
│       ├── generate_visualizations.py   # Generate DAGs for all players (all cycles)
│       └── generate_raw_data_dict.py    # Generate data/raw/Raw Data Dictionary.pdf
│
└── results/                         # Model outputs and saved figures
```

---

## Data Pipeline

### Data Asset Overview

The project draws on **4 raw Excel datasets** from OH Leuven's player monitoring system. The preprocessing pipeline uses Readiness_Data as the base dataset, merging in Raw_Data, Sessions, and Games.

| Dataset | Rows | Columns | Players | Date Range | Granularity |
|---------|------|---------|---------|------------|-------------|
| **Readiness_Data.xlsx** | 14,359 | 24 | 28 | 2024-07-01 → 2026-02-17 (597 days) | Daily (player-day) |
| **Raw_Data.xlsx** | 9,968 | 39 | 84 | 2024-05-02 → 2026-03-01 | Session-level (player-session) |
| **Sessions.xlsx** | 1,206 | 9 | — | 2024-05-02 → 2026-03-01 | Session-level (team-session) |
| **Games.xlsx** | 403 | 9 | 24 | 2025-07-27 → 2026-02-28 (216 days, 27 match dates) | Match-level (player-match) |

**Processed dataset** (auto-generated from all datasets): 14,359 rows × 46 columns → `data/processed/RTT.xlsx`

#### Player Overlap Across Datasets

- Readiness_Data ∩ Raw_Data: **28** (all Readiness_Data players appear in Raw_Data)
- Raw_Data has 84 total players (56 not in Readiness_Data)
- Games has 24 players; overlap with Readiness_Data: **23** players
- All 3 player-level datasets overlap: **23** players

### Raw Dataset Schemas

#### Readiness_Data.xlsx (24 columns)

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

#### Raw_Data.xlsx (39 columns)

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

#### Sessions.xlsx (9 columns)

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

#### Games.xlsx (9 columns)

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
| External Load (GPS) | Total Distance, High-Speed Distance, Decelerations, Sprints | ACWR (7:42 EMA), % of personal benchmarks | Readiness_Data |
| Training Intensity | Training Intensity Yesterday | Composite: tanh(mean(TD%, HSD%, Dec%, Sprints%) / 100). Soft cap, range [0, 1) | Engineered from GPS % |
| Raw GPS/HR (Yesterday) | Total Minutes, Total Distance (m), High Speed Distance (m), Avg Heart Rate, Heart Rate Exertion | Absolute values, shifted +1 day | Raw_Data |
| Match Performance (Yesterday) | Match HID Per BIP, Match HIE Per BIP, Match Minutes Played | Continuous, only filled day after match | Games |
| Match Intensity (Yesterday) | Match Intensity Yesterday | Causal outcome Y: geometric mean of HID and HIE Per BIP × min(minutes, 60)/60. Range ≈ [0, ∞) | Engineered from Games |
| Subjective Wellbeing | Fatigue, Soreness, Sleep Quality, Stress, Mood | Individualised 28-day rolling-window Z-scores | Readiness_Data |
| Composite Scores | Physical State, Mental State, Overall Wellbeing | Aggregated from subjective sub-scales | Engineered |
| Contextual / Medical | Medical availability %, Club attendance %, Activity reason, Medical status | Categorical | Readiness_Data |
| Temporal Context | Days Since Game, Days Until Match | Integer; Days Since Game ≥ 1, Days Until Match ≥ 0 |

### 1. Data Preprocessing (`data_preprocessing.py`)

The preprocessing pipeline merges all raw datasets into a single analysis-ready file.

**Input:** `data/raw/Readiness_Data.xlsx`, `Raw_Data.xlsx`, `Sessions.xlsx`, `Games.xlsx`
**Output:** `data/processed/RTT.xlsx` + `RTT Data Dictionary.pdf`

**Transformations:**
1. Load all raw xlsx files (Readiness_Data as base, Raw_Data, Sessions, Games)
2. Player ID mapping (complex keys → sequential IDs 1-28)
3. Column renaming for clarity
4. Percentage column cleaning (string → integer)
5. Comment categorization (recovery, discomfort, stiffness, etc.)
6. **Fill Activity Type Yesterday from Raw_Data** — for rows where Readiness_Data's `Reason` field is blank, look up the player's entry in Raw_Data on the same date (t-1) and use that `Reason` value (modal value across drill records).
7. **Fill free days** — rows where Activity Type Yesterday is still NaN AND all GPS % columns are also NaN are true rest days. Set Activity Type Yesterday = `'Free'`, GPS % columns = `0`. Training Intensity Yesterday will then be `tanh(0) = 0`. Rows where Activity Type is NaN but GPS % data exists are data-entry artefacts and are left as-is.
8. **Merge Raw_Data** GPS/HR columns (total_minutes, total_distance, high_speed_distance, avg_heartrate, heart_rate_exertion) — aggregated per player-day (sum for volume, weighted mean for HR), shifted +1 day so day-of data becomes "yesterday"
9. **Merge Games** match performance columns (High Intensity Per BIP, HIT Efforts Per BIP, minutes_played) — shifted +1 day (match data → day after match)
10. Composite scores (Physical State, Mental State, Overall Wellbeing)
11. Activity Type Today (session type on day t, derived from next row's Activity Type Yesterday)
12. Days Since Game (days since last *completed* match, minimum 1, never 0)
13. Days Until Match (days until next scheduled match, 0 on match day)
14. Match Day (team-level) and Selected (player-level)
15. Status Decrease detection
16. ACWR danger zone flagging (any ACWR > 1.5)
17. Training Intensity Yesterday composite (tanh(mean(TD%, HSD%, Dec%, Sprints%) / 100), soft cap in [0, 1)); free days yield 0.0
18. Column reordering into temporal groups
19. Save RTT.xlsx + auto-generate PDF data dictionary

**Processed Dataset Columns (46 columns, in order):**

| Group | Columns | Temporal Position | Source |
|-------|---------|-------------------|--------|
| Identifiers | Date, Playerkey, Player ID, Position | Always known | RD |
| Historical | Medical Availability Last 14 Days, Club Attendance Last 14 Days | Before day t | RD |
| Yesterday (t-1) RD | ACWR (×4), Any ACWR Danger, Activity Type Yesterday, Comment Yesterday, Comment Category Yesterday, GPS % (×5), Training Intensity Yesterday, Perceived Exertion Yesterday | Before day t | RD |
| Yesterday (t-1) Raw | Total Minutes, Total Distance (m), High Speed Distance (m), Avg Heart Rate, Heart Rate Exertion | Before day t | Raw_Data (shifted) |
| Yesterday (t-1) Games | Match HID Per BIP, Match HIE Per BIP, Match Minutes Played, **Match Intensity** | Before day t (only day after match) | Games (shifted) + Engineered |
| Morning (t) | Status, Status Decrease, Fatigue/Readiness/Soreness (z), Physical State, Sleep Quality/Stress/Mood (z), Mental State, Overall Wellbeing, Days Since Game, Days Until Match, Match Day | Covariates Lₜ (Match Day = schedule info, known in advance) | RD + Engineered |
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

### Method 1: Linear Regression (`src/models/lin_reg.py`)

**Best For:** Interpretability, regression targets (continuous outcomes), fast baseline with L2 regularisation

Ridge regression (L2-regularised) for continuous targets; `RidgeClassifier` as automatic fallback for binary/multi-class targets. Regularisation strength `alpha` is optimised via Optuna. Requires standardisation because the Ridge penalty is scale-sensitive.

**Key Parameters:**
```python
LinearRegressionModel(
    target_variable='Match Intensity Yesterday',
    predictory_columns=[...],
    lag=3,
    hpo_trials=20,                 # Optuna trials for alpha (1e-4 → 1e3, log scale)
    alpha=1.0,                     # Used when hpo_trials=0
    categorical_encoding='one-hot', # Recommended for linear models
    standardize=True               # Required — Ridge is scale-sensitive
)
```

### Method 2: Logistic Regression (`src/models/log_reg.py`)

**Best For:** Interpretability, baseline performance, feature importance analysis

**Key Parameters:**
```python
LogisticRegressionModel(
    target_variable='Status Decrease',
    predictory_columns=[...],
    lag=3,
    hpo_trials=20,                # Optuna hyperparameter optimization
    class_weight='balanced',      # Auto-balance for imbalanced data
    categorical_encoding='one-hot',
    standardize=True              # Required for linear models
)
```

### Method 3: XGBoost (`src/models/xgboost.py`)

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

### Method 4: CatBoost (`src/models/catboost.py`)

**Best For:** Native categorical feature handling, robust gradient boosting with automatic overfitting detection

**Key Parameters:**
```python
CatBoostModel(
    target_variable='Status Decrease',
    predictory_columns=[...],
    lag=3,
    hpo_trials=20,                # Optuna hyperparameter optimization
    iterations=200,
    depth=6,
    learning_rate=0.1,
    categorical_encoding='label', # CatBoost handles categoricals natively
    standardize=False             # Trees don't need standardization
)
```

### Method 5: TabPFN (`src/models/tabpfn.py`)

**Best For:** Small tabular datasets, zero-shot classification, no hyperparameter tuning required

TabPFN (Prior-Data Fitted Networks) is a transformer pre-trained on synthetic tabular datasets. It performs **in-context learning** — no iterative training at fit time. The model infers predictions directly from the training set, making it extremely fast and often competitive on small datasets.

**Key Parameters:**
```python
TabPFNModel(
    target_variable='Status Decrease',
    predictory_columns=[...],
    lag=3,
    device='auto',                # 'auto', 'cpu', or 'cuda'
    n_estimators=4,               # Number of ensemble members
    categorical_encoding='label', # Label encoding for categorical features
    standardize=False             # TabPFN handles its own normalization
)
```

**Note:** TabPFN supports both classification (`TabPFNClassifier`) and regression (`TabPFNRegressor`) — task type is auto-detected from the target. No HPO is performed (hyperparameters are fixed by the pre-trained model). Install with: `pip install tabpfn`

All five models support `target_horizon` for predicting future outcomes:
```python
model = XGBoostModel(
    target_horizon=1,  # Predict tomorrow's value using today's features
    ...
)
```

### Causal DAG Builder (`dag_creator.py`)

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
from src.methods.dag_creator import DAGCreator

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

Only 28 players but moderately long time series (~597 days total). The signal is likely dominated by **within-player dynamics**. Models must be parsimonious or leverage partial pooling to avoid overfitting. Methods should balance individual-level tailoring with limited sample size.

---

## Usage Guide

### Quick Start

```python
from src.models.xgboost import XGBoostModel

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
    'Match HID Per BIP Yesterday',
    'Match HIE Per BIP Yesterday',
    'Match Minutes Played Yesterday',
    'Match Intensity Yesterday',   # causal outcome Y for DTR experiments
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

**Naming convention:** `0.` = debugging / temporary, `1.x` = data visualisation, `2.x` / `3.x` = experiments

| Notebook | Purpose |
|----------|---------|
| `1.1. Match Analysis.ipynb` | Match-level data exploration: match intensity distributions, per-player profiles, playing time patterns, match performance metrics. |
| `0. Processed_Data_Quality.ipynb` | Automated quality checks on RTT.xlsx: player coverage, column completeness, temporal integrity, ACWR flags, encoding validation. |
| `0. TI_Missingness_Analysis.ipynb` | Missingness analysis for Training Intensity Yesterday; investigates free-day fill logic and NaN patterns. |
| `1.2. Raw Data Visualisation.ipynb` | Comprehensive EDA across all 4 raw datasets: missingness, dataset linkage (Venn diagrams), temporal coverage, wellness/GPS distributions, player profiles (radar charts), cross-dataset correlations. |
| `1.3. Processed Data Visualisation.ipynb` | EDA of processed RTT.xlsx: variable distributions, temporal patterns, per-player profiles, correlation heatmaps. |
| `2.1. Experiment1.ipynb` | **Reference only.** Predicts Match Intensity (largely unidentifiable causal target). Retained for documentation purposes. |
| `2.2. Experiment2.ipynb` | **Treatment policy modelling.** Predicts today's Training Intensity from morning covariates. Runs lin_reg, XGBoost, CatBoost, TabPFN; compares RMSE/R²; lag ablation; feature importances. Propensity model foundation. |
| `2.3. Experiment3.ipynb` | **Short-term load response.** Predicts next-day Status Decrease (binary). Prediction mode only. ROC/PR curves, per-player AUC. Models: LogReg, XGBoost, CatBoost, TabPFN. |

### Running from Command Line

```bash
cd "path/to/Readiness-To-Train"
python src/data/data_preprocessing.py         # Run multi-dataset preprocessing -> RTT.xlsx
python src/utils/generate_raw_data_dict.py      # Regenerate data/raw/Raw Data Dictionary.pdf
python src/models/log_reg.py                  # Run Logistic Regression
python src/models/xgboost.py                  # Run XGBoost
python src/models/catboost.py                 # Run CatBoost
python src/models/tabpfn.py                   # Run TabPFN
python src/methods/dag_creator.py             # Run DAG demos + generate visualizations
python src/utils/generate_project_overview.py # Regenerate Project Overview.pdf
python src/utils/generate_raw_data_dict.py    # Regenerate data/raw/Raw Data Dictionary.pdf
python scripts/Experiment1.py                 # Exp 1 demo — Match Intensity (reference only)
python scripts/Experiment2.py                 # Exp 2 demo — Training Intensity prediction (XGBoost, lag=3)
python scripts/Experiment3.py                 # Exp 3 demo — Status Decrease prediction (both modes)
python src/utils/generate_visualizations.py   # Generate DAGs for all players
python src/utils/generate_project_results.py # Generate Project Results.pdf
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

`LinearRegressionModel`, `LogisticRegressionModel`, `XGBoostModel`, `CatBoostModel`, and `TabPFNModel` all share:
```python
model.train() -> Dict  # Returns predictions, metrics, model weights
```

### Experiment 1 Runner (`scripts/Experiment1.py`)

> **Status:** Reference / exploratory only. Predicts Match Intensity Yesterday — a causal target that is largely unidentifiable from available data due to unobserved confounders between training load and match performance. Retained for reference; primary experiments are Exp 2 and Exp 3.

Regression runner for `'lin_reg'`, `'xgboost'`, `'catboost'`, `'tabpfn'`. Returns `metrics` dict with mse, rmse, mae, r2, pearson_r/p, null_rmse, train_rmse, n_test, n_train.

### Experiment 2 Runner (`scripts/Experiment2.py`)

**Treatment Policy Modelling** — predict today's Training Intensity from morning covariates.

```python
from scripts.Experiment2 import run_experiment, DEFAULT_COVARIATES

results = run_experiment(
    covariates=DEFAULT_COVARIATES,   # or custom list
    lag=3,
    model_type='xgboost',   # 'lin_reg', 'xgboost', 'catboost', 'tabpfn'
)
```

**Returns:** Same structure as Exp 1 (regression metrics). High R² → coaching policy is systematic; low R² → unmeasured factors dominate. Either way, the fitted model is a valid propensity model for IPTW.

### Experiment 3 Runner (`scripts/Experiment3.py`)

**Short-term Load Response** — predict next-day Status Decrease (binary).

```python
from scripts.Experiment3 import run_experiment, DEFAULT_COVARIATES

# Pure prediction
results = run_experiment(
    covariates=DEFAULT_COVARIATES, lag=3, model_type='xgboost',
    mode='prediction',
)

# Causal framing: adds Training Intensity as treatment (diagnostic, NOT valid causal estimate)
results = run_experiment(
    covariates=DEFAULT_COVARIATES, lag=3, model_type='log_reg',
    mode='prediction',
)
```

**Parameters:** `covariates`, `lag` (>= 0), `model_type` (`'log_reg'`, `'xgboost'`, `'catboost'`, `'tabpfn'`), `mode` (`'prediction'`), `test_size`, `val_size`, `random_state`, `verbose`, `**model_kwargs`.

**Returns:**
```python
{
    'config':        dict,
    'y_train_true':  ndarray,
    'y_train_pred':  ndarray,   # probabilities
    'y_test_true':   ndarray,
    'y_test_pred':   ndarray,   # probabilities
    'meta_test':     pd.DataFrame,
    'metrics':       dict,   # roc_auc, avg_precision, f1, precision, recall,
                             # accuracy, prevalence, null_accuracy, threshold,
                             # n_test, n_train, n_positive_test
    'per_player':    dict,   # Player ID → {n, n_positive, roc_auc, f1}
    'model_weights': dict,
    'feature_names': list,
    'best_params':   dict,
    'task_type':     str,    # 'classification'
    'threshold':     float,
}
```

### DAG Batch Generator (`src/utils/generate_visualizations.py`)

```python
from src.utils.generate_visualizations import generate_all_player_dags
generate_all_player_dags()  # Generates 4 DAGs per player into images/DAGs/player &lt;id&gt;/
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
- All "Yesterday" columns from Readiness_Data (ACWR, GPS %, Training Intensity Yesterday, RPE, comments)
- All "Yesterday" columns from Raw_Data (Total Minutes, Total Distance (m), High Speed Distance (m), Avg Heart Rate, Heart Rate Exertion)
- All "Yesterday" columns from Games (Match HID Per BIP, Match HIE Per BIP, Match Minutes Played) — only filled day after match
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

| Scenario | target | treatment_columns | treatment_horizon | target_horizon | Experiment |
|----------|--------|------------------|-------------------|----------------|------------|
| Propensity model (Exp 2) | `Training Intensity Yesterday` | *not set* | - | 1 | Exp 2 |
| Load response — prediction (Exp 3) | `Status Decrease` | *not set* | - | 1 | Exp 3 |
| Load response — causal framing (Exp 3) | `Status Decrease` | `['Training Intensity Yesterday']` | 1 | 1 | Exp 3 |
| Session-type effect | `Status Decrease` | `['Activity Type Today']` | 0 | 1 | Custom |
| Full DTR prescription | `Status Decrease` | `['Activity Type Today', 'Training Intensity Yesterday']` | `{...: 0, ...: 1}` | 1 | Custom |

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

**Last Updated:** 2026-04-16
**Python Version:** 3.8+
**Key Dependencies:** torch (CUDA build), pandas (<3.0), numpy, scipy, scikit-learn, xgboost, catboost, tabpfn, optuna, shap, tqdm, Pillow, matplotlib, seaborn, missingno, matplotlib-venn, plotly, reportlab, openpyxl, networkx, jinja2
