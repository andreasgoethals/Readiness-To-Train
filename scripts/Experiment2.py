"""
Experiment 2: Treatment Policy Modelling
=========================================

Research question:
    What morning-state features predict the training intensity assigned by
    the coaching staff?

This experiment models the **propensity / treatment-assignment policy**:

    pi(A_t | L_t, history)

where A_t is today's training intensity (continuous, [0, 1)) and L_t is the
player's morning covariate state at time t.  Understanding this policy has
two uses:
  1. Interpretability — which player-state signals most strongly drive
     coaching load decisions?
  2. Causal foundation — the propensity model is required for IPTW-based
     marginal structural models and for doubly-robust estimators.

Temporal alignment
------------------
target_variable  = 'Training Intensity Yesterday'
target_horizon   = 1

The target is "Training Intensity Yesterday" shifted one step forward, so
the predicted value is today's training intensity (assigned AFTER the morning
assessment) using only morning covariates that were observed BEFORE the
session began.

Models
------
'lin_reg', 'xgboost', 'catboost', 'tabpfn'

Usage
-----
    from scripts.Experiment2 import run_experiment

    results = run_experiment(
        covariates=['Fatigue (z)', 'Readiness (z)', 'Days Until Match', ...],
        lag=3,
        model_type='xgboost',
    )
    print(results['metrics'])
"""

import sys
import inspect
import traceback
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.lin_reg import LinearRegressionModel
from models.xgboost import XGBoostModel
from models.catboost import CatBoostModel
from models.tabpfn import TabPFNModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = 'Training Intensity Yesterday'

VALID_MODELS = {'lin_reg', 'xgboost', 'catboost', 'tabpfn'}

MODEL_DEFAULTS = {
    'lin_reg': {
        'categorical_encoding': 'one-hot',
        'standardize': True,
        'hpo_trials': 20,
        'max_iter': 1000,
    },
    'xgboost': {
        'categorical_encoding': 'label',
        'standardize': False,
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'early_stopping_rounds': 10,
        'device': 'cuda',        # GPU acceleration (XGBoost >= 2.0, CUDA required)
    },
    'catboost': {
        'categorical_encoding': 'label',
        'standardize': False,
        'hpo_trials': 10,
        'iterations': 200,
        'depth': 6,
        'learning_rate': 0.1,
        'task_type': 'GPU',      # GPU acceleration (CatBoost native CUDA support)
    },
    'tabpfn': {
        'categorical_encoding': 'label',
        'standardize': False,
        'n_estimators': 4,
        'device': 'cuda',                    # GPU inference
        'ignore_pretraining_limits': True,   # override 1000-sample CPU limit
    },
}


# ---------------------------------------------------------------------------
# Default covariate set
# ---------------------------------------------------------------------------

# All columns that are known at morning-assessment time (before the session).
# Activity Type Yesterday is included because the previous session type is
# a strong predictor of the next session type (periodisation logic).
DEFAULT_COVARIATES = [
    'Physical State',
    'Mental State',
    'Total Distance (ACWR) Yesterday',
    'High Speed Distance (ACWR) Yesterday',
    'Any ACWR Danger',
    'Total Distance % Yesterday',
    'High Speed Distance % Yesterday',
    'Perceived Exertion Yesterday',
    'Total Minutes Yesterday',
    'Avg Heart Rate Yesterday',
    'Heart Rate Exertion Yesterday',
    'Days Since Game',
    'Days Until Match',
    'Player ID',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Return only the kwargs accepted by cls.__init__, silently dropping others."""
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {'self'}
    return {k: v for k, v in kwargs.items() if k in accepted}


def _build_model(model_type: str, covariates: List[str], lag: int,
                 test_size: float, val_size: float, random_state: int,
                 **kwargs):
    """Instantiate the requested model with merged defaults + overrides.

    Uses inspect.signature to filter **common to only kwargs accepted by
    each constructor — makes the function robust to GPU param mismatches
    (e.g. device='cuda', task_type='GPU') and stale module imports.
    """
    params = MODEL_DEFAULTS[model_type].copy()
    params.update(kwargs)

    common = dict(
        target_variable=TARGET,
        predictory_columns=covariates,
        lag=lag,
        include_previous_target=False,
        target_horizon=1,          # predict TODAY's intensity (assigned post-assessment)
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        missing_numeric='mean',
        missing_categorical='mode',
    )
    common.update(params)

    cls_map = {
        'lin_reg':  LinearRegressionModel,
        'xgboost':  XGBoostModel,
        'catboost': CatBoostModel,
        'tabpfn':   TabPFNModel,
    }
    if model_type not in cls_map:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    cls = cls_map[model_type]
    return cls(**_filter_kwargs(cls, common))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_train_true: np.ndarray, y_train_pred: np.ndarray) -> Dict:
    """Regression metrics for the test set, with null baseline and train RMSE."""
    residuals = y_true - y_pred
    mse  = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r_val, p_val = stats.pearsonr(y_true, y_pred)
    else:
        r_val, p_val = 0.0, 1.0

    null_pred  = np.full(len(y_true), np.mean(y_train_true))
    null_rmse  = float(np.sqrt(np.mean((y_true - null_pred) ** 2)))
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


def _per_player_metrics(y_true: np.ndarray, y_pred: np.ndarray, meta) -> Dict:
    """Per-player RMSE and R² on the test set.

    Falls back to 'Playerkey' if 'Player ID' is not in meta (can happen
    when 'Player ID' is in the covariate list and gets one-hot encoded,
    removing the original column from the processed DataFrame).
    """
    pid_col = next((c for c in ('Player ID', 'Playerkey') if c in meta.columns), None)
    if pid_col is None:
        return {}
    per_player = {}
    player_ids = meta[pid_col].values
    for pid in np.unique(player_ids):
        mask = player_ids == pid
        n = int(mask.sum())
        if n < 2:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
        per_player[pid] = {'n': n, 'rmse': rmse, 'r2': r2}
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
    Run Experiment 2: predict today's Training Intensity from morning covariates.

    Parameters
    ----------
    covariates : list of str
        Covariate columns to use as predictors.  Only columns observable at
        morning-assessment time should be included (see CLAUDE.md §Temporal
        Semantics).
    lag : int
        Number of lag steps, must be >= 1.  With lag=1 only row-t covariates
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
        Override specific model hyperparameters, e.g. hpo_trials=10.

    Returns
    -------
    dict with keys:

        config          dict   experiment configuration
        y_train_true    ndarray
        y_train_pred    ndarray
        y_val_true      ndarray  (only present when val set is non-empty)
        y_val_pred      ndarray  (only present when val set is non-empty)
        y_test_true     ndarray
        y_test_pred     ndarray
        meta_test       pd.DataFrame   Date + Player ID for each test row
        metrics         dict   mse, rmse, mae, r2, pearson_r/p,
                               null_rmse, train_rmse, n_test, n_train
        per_player      dict   Player ID -> {n, rmse, r2}
        model_weights   dict   coefficients / importances + feature_names
        feature_names   list of str
        best_params     dict
        task_type       str   ('regression')

    Notes
    -----
    - High R² here means the model can recover the coach's load policy from
      observable player state — useful for understanding decision-making.
    - Low R² does NOT necessarily mean the experiment failed; coaches may use
      information not captured in these covariates (tactical, motivational).
    - The fitted model can be repurposed as a propensity model for IPTW.
    """
    if model_type not in VALID_MODELS:
        raise ValueError(
            f"model_type must be one of {sorted(VALID_MODELS)}, got {model_type!r}"
        )
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if not covariates:
        raise ValueError("covariates list must not be empty")

    if verbose:
        print("=" * 72)
        print("Experiment 2: Treatment Policy Modelling")
        print("Predict today's Training Intensity from morning covariates")
        print("=" * 72)
        print(f"  Target:      {TARGET}  (target_horizon=1)")
        print(f"  Model:       {model_type}")
        print(f"  Lag:         {lag}")
        print(f"  Covariates:  {len(covariates)}")
        for c in covariates:
            print(f"               . {c}")
        print(f"  Test size:   {test_size}")
        print(f"  Val size:    {val_size}")
        if model_kwargs:
            print(f"  Overrides:   {model_kwargs}")
        print()

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

    metrics = _compute_metrics(
        y_true=raw['y_test_true'],
        y_pred=raw['y_test_pred'],
        y_train_true=raw['y_train_true'],
        y_train_pred=raw['y_train_pred'],
    )

    per_player: Dict = {}
    meta_test = model.data.get('meta_test')
    if meta_test is not None and not meta_test.empty:
        per_player = _per_player_metrics(
            raw['y_test_true'], raw['y_test_pred'], meta_test
        )

    if verbose:
        print(f"  Test  RMSE:  {metrics['rmse']:.4f}  (null: {metrics['null_rmse']:.4f})")
        print(f"  Train RMSE:  {metrics['train_rmse']:.4f}")
        print(f"  R2:          {metrics['r2']:.4f}")
        print(f"  Pearson r:   {metrics['pearson_r']:.4f}  (p = {metrics['pearson_p']:.4f})")
        print(f"  MAE:         {metrics['mae']:.4f}")
        print(f"  N test rows: {metrics['n_test']}  |  N train rows: {metrics['n_train']}")
        print()

    result = {
        'config': {
            'experiment':     'Experiment2_TreatmentPolicy',
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
        # Expose fitted estimator and split data for downstream analysis (e.g. SHAP)
        'trained_model': raw.get('trained_model'),
        'X_train':       model.data.get('X_train'),
        'X_test':        model.data.get('X_test'),
    }

    if 'y_val_true' in raw:
        result['y_val_true'] = raw['y_val_true']
        result['y_val_pred'] = raw['y_val_pred']

    if meta_test is not None:
        result['meta_test'] = meta_test

    return result


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = run_experiment(
        covariates=DEFAULT_COVARIATES,
        lag=3,
        model_type='xgboost',
    )

    print(f"Final: RMSE={results['metrics']['rmse']:.4f}  "
          f"(null={results['metrics']['null_rmse']:.4f})  "
          f"R2={results['metrics']['r2']:.4f}")
