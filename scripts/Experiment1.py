"""
Experiment 1: Predict Match Intensity Using ML Models
=====================================================

Predicts the match intensity of the next match for each player.  Given
covariates observed at time t (morning assessment + yesterday's load), the
model predicts the Match Intensity that will be recorded at t+1, i.e.
'Match Intensity Yesterday' in the next row (target_horizon = 1).

This is the first step of causal analysis: a pure predictive baseline before
introducing causal/DTR methods.

Target : 'Match Intensity Yesterday'  (continuous, range ≈ 0–1)
Models : 'lin_reg', 'xgboost', 'catboost', 'tabpfn'

Lag semantics
-------------
lag = 1  →  use covariates from the current row (t) only to predict MI at t+1.
lag = 3  →  use covariates from rows t, t-1, t-2 (lagged features created
             per-player so no cross-player leakage).

Usage
-----
    from scripts.Experiment1 import run_experiment

    results = run_experiment(
        covariates=['Fatigue (z)', 'Readiness (z)', 'Days Until Match', ...],
        lag=3,
        model_type='xgboost',
    )
    print(results['metrics'])
"""

import sys
import traceback
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from scipy import stats

# Add src/ to path so model imports resolve
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.lin_reg import LinearRegressionModel
from models.xgboost import XGBoostModel
from models.catboost import CatBoostModel
from models.tabpfn import TabPFNModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = 'Match Intensity Yesterday'

VALID_MODELS = {'lin_reg', 'xgboost', 'catboost', 'tabpfn'}

# Default hyperparameters for each model type.
# These can be overridden via **model_kwargs in run_experiment().
MODEL_DEFAULTS = {
    'lin_reg': {
        'categorical_encoding': 'one-hot',
        'standardize': True,
        'hpo_trials': 50,
        'max_iter': 1000,
    },
    'xgboost': {
        'categorical_encoding': 'label',
        'standardize': False,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'early_stopping_rounds': 10,
    },
    'catboost': {
        'categorical_encoding': 'label',
        'standardize': False,
        'hpo_trials': 50,
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.1,
    },
    'tabpfn': {
        'categorical_encoding': 'label',
        'standardize': False,
        'n_estimators': 4,
        'device': 'auto',
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model(model_type: str, covariates: List[str], lag: int,
                 test_size: float, val_size: float, random_state: int,
                 **kwargs):
    """Instantiate the requested model with merged defaults + overrides."""
    params = MODEL_DEFAULTS[model_type].copy()
    params.update(kwargs)

    common = dict(
        target_variable=TARGET,
        predictory_columns=covariates,
        lag=lag,
        include_previous_target=False,
        target_horizon=1,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        missing_numeric='mean',
        missing_categorical='mode',
    )
    common.update(params)

    if model_type == 'lin_reg':
        return LinearRegressionModel(**common)
    elif model_type == 'xgboost':
        return XGBoostModel(**common)
    elif model_type == 'catboost':
        return CatBoostModel(**common)
    elif model_type == 'tabpfn':
        return TabPFNModel(**common)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_train_true: np.ndarray, y_train_pred: np.ndarray) -> Dict:
    """
    Compute regression metrics for the test set, including null baseline and
    train-set RMSE for overfitting diagnostics.
    """
    # Core test metrics
    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Pearson correlation
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r_val, p_val = stats.pearsonr(y_true, y_pred)
    else:
        r_val, p_val = 0.0, 1.0

    # Null baseline: always predict the training-set mean
    null_pred = np.full(len(y_true), np.mean(y_train_true))
    null_rmse = float(np.sqrt(np.mean((y_true - null_pred) ** 2)))

    # Training RMSE (overfitting check)
    train_rmse = float(np.sqrt(np.mean((y_train_true - y_train_pred) ** 2)))

    return {
        'mse':        mse,
        'rmse':       rmse,
        'mae':        mae,
        'r2':         r2,
        'pearson_r':  float(r_val),
        'pearson_p':  float(p_val),
        'null_rmse':  null_rmse,
        'train_rmse': train_rmse,
        'n_test':     int(len(y_true)),
        'n_train':    int(len(y_train_true)),
    }


def _per_player_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         meta) -> Dict:
    """
    Compute per-player RMSE and R² on the test set.

    Parameters
    ----------
    meta : pd.DataFrame
        Must contain a 'Player ID' column aligned with y_true / y_pred.
    """
    per_player = {}
    player_ids = meta['Player ID'].values
    for pid in np.unique(player_ids):
        mask = player_ids == pid
        n = int(mask.sum())
        if n < 2:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
        per_player[int(pid)] = {'n': n, 'rmse': rmse, 'r2': r2}
    return per_player


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(
    covariates: List[str],
    lag: int,
    model_type: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
    **model_kwargs,
) -> Dict:
    """
    Run Experiment 1: predict Match Intensity Yesterday (t+1) for each player.

    Parameters
    ----------
    covariates : list of str
        Covariate columns to use as predictors.  Only columns available at
        morning-assessment time (before any training session) should be
        included (see CLAUDE.md Temporal Semantics).
    lag : int
        Number of lag steps, must be ≥ 1.  With lag=1 only row t covariates
        are used; with lag=3 rows t, t-1, t-2 are used (per-player grouping
        ensures no cross-player leakage).
    model_type : str
        One of 'lin_reg', 'xgboost', 'catboost', 'tabpfn'.
    test_size : float, default=0.2
        Fraction of dates reserved for test (temporal: latest dates).
    val_size : float, default=0.1
        Fraction of dates reserved for validation.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress and key metrics to stdout.
    **model_kwargs
        Override specific model hyperparameters, e.g. ``hpo_trials=10``.

    Returns
    -------
    dict with keys:

        config          dict  experiment configuration
        y_train_true    ndarray
        y_train_pred    ndarray
        y_val_true      ndarray  (only present when val set is non-empty)
        y_val_pred      ndarray  (only present when val set is non-empty)
        y_test_true     ndarray
        y_test_pred     ndarray
        meta_test       pd.DataFrame  Date + Player ID for each test row
        metrics         dict  MSE, RMSE, MAE, R², pearson_r/p,
                              null_rmse, train_rmse, n_test, n_train
        per_player      dict  Player ID → {n, rmse, r2}
        model_weights   dict  coefficients / importances + feature_names
        feature_names   list of str
        best_params     dict
        task_type       str  ('regression')
    """
    # ── validate inputs ──────────────────────────────────────────────────────
    if model_type not in VALID_MODELS:
        raise ValueError(
            f"model_type must be one of {sorted(VALID_MODELS)}, got '{model_type}'"
        )
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    if not covariates:
        raise ValueError("covariates list must not be empty")

    # ── log configuration ─────────────────────────────────────────────────────
    if verbose:
        print("=" * 72)
        print("Experiment 1: Predict Match Intensity Yesterday (t+1)")
        print("=" * 72)
        print(f"  Model:       {model_type}")
        print(f"  Lag:         {lag}")
        print(f"  Covariates:  {len(covariates)}")
        for c in covariates:
            print(f"               · {c}")
        print(f"  Test size:   {test_size}")
        print(f"  Val size:    {val_size}")
        if model_kwargs:
            print(f"  Overrides:   {model_kwargs}")
        print()

    # ── build and train model ─────────────────────────────────────────────────
    model = _build_model(
        model_type=model_type,
        covariates=covariates,
        lag=lag,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        **model_kwargs,
    )
    raw = model.train()

    # ── metrics ───────────────────────────────────────────────────────────────
    metrics = _compute_metrics(
        y_true=raw['y_test_true'],
        y_pred=raw['y_test_pred'],
        y_train_true=raw['y_train_true'],
        y_train_pred=raw['y_train_pred'],
    )

    # ── per-player breakdown ──────────────────────────────────────────────────
    per_player: Dict = {}
    meta_test = model.data.get('meta_test')
    if meta_test is not None and not meta_test.empty:
        per_player = _per_player_metrics(
            raw['y_test_true'], raw['y_test_pred'], meta_test
        )

    if verbose:
        print(f"  Test  RMSE:  {metrics['rmse']:.4f}  (null: {metrics['null_rmse']:.4f})")
        print(f"  Train RMSE:  {metrics['train_rmse']:.4f}")
        print(f"  R²:          {metrics['r2']:.4f}")
        print(f"  Pearson r:   {metrics['pearson_r']:.4f}  (p = {metrics['pearson_p']:.4f})")
        print(f"  MAE:         {metrics['mae']:.4f}")
        print(f"  N test rows: {metrics['n_test']}  |  N train rows: {metrics['n_train']}")
        print()

    # ── assemble result dict ──────────────────────────────────────────────────
    result = {
        'config': {
            'covariates':     covariates,
            'lag':            lag,
            'model_type':     model_type,
            'target':         TARGET,
            'target_horizon': 1,
            'test_size':      test_size,
            'val_size':       val_size,
            'random_state':   random_state,
            'model_kwargs':   model_kwargs,
        },
        'y_train_true':  raw['y_train_true'],
        'y_train_pred':  raw['y_train_pred'],
        'y_test_true':   raw['y_test_true'],
        'y_test_pred':   raw['y_test_pred'],
        'metrics':       metrics,
        'per_player':    per_player,
        'model_weights': raw.get('model_weights', {}),
        'feature_names': model.data.get('feature_names', []),
        'best_params':   raw.get('best_params', {}),
        'task_type':     raw.get('task_type', 'regression'),
    }

    # optional: val predictions (only when val set is non-empty)
    if 'y_val_true' in raw:
        result['y_val_true'] = raw['y_val_true']
        result['y_val_pred'] = raw['y_val_pred']

    # meta for test rows (Date + Player ID)
    if meta_test is not None:
        result['meta_test'] = meta_test

    return result


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    predictors = [
        # Morning wellness assessment
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
        'Physical State', 'Sleep Quality (z)', 'Stress (z)',
        'Mood (z)', 'Mental State', 'Overall Wellbeing',
        # Yesterday's workload
        'Total Distance (ACWR) Yesterday',
        'High Speed Distance (ACWR) Yesterday',
        'Any ACWR Danger',
        'Training Intensity Yesterday',
        # Temporal context
        'Days Since Game', 'Days Until Match',
        # Historical context
        'Medical Availability Last 14 Days',
        'Club Attendance Last 14 Days',
        # Player profile + previous session type
        'Position', 'Activity Type Yesterday',
    ]

    results = run_experiment(
        covariates=predictors,
        lag=3,
        model_type='xgboost',
    )

    print(f"Final: RMSE={results['metrics']['rmse']:.4f}  "
          f"(null={results['metrics']['null_rmse']:.4f})  "
          f"R²={results['metrics']['r2']:.4f}")
