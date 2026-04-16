"""
Experiment 3: Short-term Load Response (Status Deterioration)
==============================================================

Research question:
    Does training load predict player status deterioration?

This experiment operates in two modes, controlled by the `mode` parameter:

  mode='prediction'
    Forward-looking baseline: uses morning covariates at time t (L_t) to
    predict TOMORROW's Status Decrease Y_{t+1} (target_horizon=1).
    Answers: "given this morning's wellness and workload history, is the
    player at risk of status deterioration by tomorrow?"

  mode='causal_framing'
    Contemporaneous causal diagnostic: uses morning covariates at time t
    (including Training Intensity Yesterday = A_{t-1}) to predict TODAY's
    Status Decrease Y_t (target_horizon=0).
    Answers: "does yesterday's training load explain today's status change,
    controlling for this morning's wellness state?"

    Training Intensity Yesterday is a morning covariate (fully known before
    the session) — no treatment_columns mechanism is needed.  The key
    conceptual addition is target_horizon=0 instead of 1, directly tying
    yesterday's load to today's observed status rather than projecting one
    day ahead.

    NOTE: this is still an observational association.  Healthy players tend
    to receive harder sessions (confounding by indication), so the raw
    coefficient on Training Intensity Yesterday is BIASED toward zero or
    negative.  Valid causal estimation requires G-methods (G-computation,
    IPTW/MSM — see CLAUDE.md §Methodological Framework).

Two-mode comparison is the diagnostic:
  If prediction AUC >> causal_framing AUC: morning state predicts future
    risk better than yesterday's load explains current status.
  If causal_framing AUC >> prediction AUC: workload history has a strong
    contemporaneous association with same-day status.

Temporal alignment
------------------
  mode='prediction'   : covariates at t -> Status Decrease at t+1
  mode='causal_framing': covariates at t (incl. TI Yesterday) -> Status Decrease at t

Models
------
'log_reg', 'xgboost', 'catboost'
(TabPFN works but is slow; pass model_type='tabpfn' explicitly if desired.)

Usage
-----
    from scripts.Experiment3 import run_experiment

    results_pred = run_experiment(
        covariates=['Fatigue (z)', 'Readiness (z)', 'Training Intensity Yesterday', ...],
        lag=3, model_type='xgboost', mode='prediction',
    )

    results_causal = run_experiment(
        covariates=['Fatigue (z)', 'Readiness (z)', 'Training Intensity Yesterday', ...],
        lag=3, model_type='log_reg', mode='causal_framing',
    )
"""

import sys
import inspect
import traceback
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.log_reg import LogisticRegressionModel
from models.xgboost import XGBoostModel
from models.catboost import CatBoostModel
from models.tabpfn import TabPFNModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = 'Status Decrease'

VALID_MODELS = {'log_reg', 'xgboost', 'catboost', 'tabpfn'}
VALID_MODES  = {'prediction', 'causal_framing'}

# target_horizon by mode:
#   prediction    -> 1  (predict TOMORROW's status from TODAY's morning state)
#   causal_framing -> 0 (predict TODAY's status; yesterday's load is in covariates)
MODE_HORIZON = {'prediction': 1, 'causal_framing': 0}

MODEL_DEFAULTS = {
    'log_reg': {
        'categorical_encoding': 'one-hot',
        'standardize': True,
        'hpo_trials': 20,
        'max_iter': 1000,
        'class_weight': 'balanced',
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

# Morning-assessment covariates only (available BEFORE the session).
DEFAULT_COVARIATES = [
    'Physical State',
    'Mental State',
    'Total Distance (ACWR) Yesterday',
    'High Speed Distance (ACWR) Yesterday',
    'Any ACWR Danger',
    'Total Distance % Yesterday',
    'High Speed Distance % Yesterday',
    'Training Intensity Yesterday',
    'Perceived Exertion Yesterday',
    'Total Minutes Yesterday',
    'Avg Heart Rate Yesterday',
    'Heart Rate Exertion Yesterday',
    'Days Since Game',
    'Days Until Match',
    'Position',
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
                 mode: str, test_size: float, val_size: float,
                 random_state: int, **kwargs):
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
        include_previous_target=True,
        target_horizon=MODE_HORIZON[mode],   # 1 for prediction, 0 for causal_framing
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        missing_numeric='mean',
        missing_categorical='mode',
    )
    common.update(params)

    cls_map = {
        'log_reg':  LogisticRegressionModel,
        'xgboost':  XGBoostModel,
        'catboost': CatBoostModel,
        'tabpfn':   TabPFNModel,
    }
    if model_type not in cls_map:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    cls = cls_map[model_type]
    return cls(**_filter_kwargs(cls, common))


def _compute_metrics_classification(
    y_true: np.ndarray, y_pred_proba: np.ndarray,
    y_train_true: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict:
    """
    Classification metrics for the test set.

    Parameters
    ----------
    y_true         : binary ground-truth labels
    y_pred_proba   : predicted probabilities for class 1
    y_train_true   : training labels (for null baseline)
    threshold      : decision threshold; if None, uses 0.5
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score, accuracy_score,
    )

    # Null baseline: always predict the majority class on training set
    majority_class = int(np.round(np.mean(y_train_true)))
    null_pred = np.full(len(y_true), majority_class)
    null_accuracy = float(np.mean(null_pred == y_true))

    # ROC AUC and average precision
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
        avg_prec = float(average_precision_score(y_true, y_pred_proba))
    else:
        roc_auc = 0.0
        avg_prec = 0.0

    # Hard predictions at threshold
    thr = threshold if threshold is not None else 0.5
    y_pred = (y_pred_proba >= thr).astype(int)

    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    acc  = float(accuracy_score(y_true, y_pred))

    # Prevalence context
    prevalence = float(np.mean(y_true))

    return {
        'roc_auc':        roc_auc,
        'avg_precision':  avg_prec,
        'f1':             f1,
        'precision':      prec,
        'recall':         rec,
        'accuracy':       acc,
        'threshold':      thr,
        'prevalence':     prevalence,
        'null_accuracy':  null_accuracy,
        'n_test':         int(len(y_true)),
        'n_train':        int(len(y_train_true)),
        'n_positive_test': int(np.sum(y_true)),
    }


def _per_player_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                         meta, threshold: float = 0.5) -> Dict:
    """Per-player classification metrics on the test set.

    Falls back to 'Playerkey' if 'Player ID' is not in meta (can happen
    when 'Player ID' is in the covariate list and gets one-hot encoded).
    """
    from sklearn.metrics import roc_auc_score, f1_score

    pid_col = next((c for c in ('Player ID', 'Playerkey') if c in meta.columns), None)
    if pid_col is None:
        return {}

    per_player = {}
    player_ids = meta[pid_col].values
    for pid in np.unique(player_ids):
        mask = player_ids == pid
        n = int(mask.sum())
        yt  = y_true[mask]
        yp  = y_pred_proba[mask]
        yh  = (yp >= threshold).astype(int)
        pos = int(np.sum(yt))
        if pos == 0 or pos == n:
            auc = None
        else:
            auc = float(roc_auc_score(yt, yp))
        f1 = float(f1_score(yt, yh, zero_division=0))
        per_player[pid] = {'n': n, 'n_positive': pos, 'roc_auc': auc, 'f1': f1}
    return per_player


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_experiment(
    covariates: List[str],
    lag: int,
    model_type: str,
    mode: str = 'prediction',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
    **model_kwargs,
) -> Dict:
    """
    Run Experiment 3: predict next-day Status Decrease from morning covariates.

    Parameters
    ----------
    covariates : list of str
        Covariate columns to use as predictors.  Only morning-assessment
        variables (available before the session) should be included.
    lag : int
        Number of lag steps, must be >= 1.
    model_type : str
        One of 'log_reg', 'xgboost', 'catboost', 'tabpfn'.
    mode : str, default='prediction'
        'prediction'     - standard ML, morning covariates only.
        'causal_framing' - adds today's Training Intensity as treatment;
                           produces (L_t, A_t, Y_{t+1}) triples for
                           downstream causal estimation.  Does NOT yield a
                           valid causal effect estimate on its own.
    test_size : float, default=0.2
    val_size : float, default=0.1
    random_state : int, default=42
    verbose : bool, default=True
    **model_kwargs
        Override specific model hyperparameters.

    Returns
    -------
    dict with keys:

        config          dict  experiment configuration (including mode)
        y_train_true    ndarray
        y_train_pred    ndarray   (probabilities)
        y_val_true      ndarray   (only when val set is non-empty)
        y_val_pred      ndarray
        y_test_true     ndarray
        y_test_pred     ndarray   (probabilities)
        meta_test       pd.DataFrame  Date + Player ID for each test row
        metrics         dict  roc_auc, avg_precision, f1, precision, recall,
                              accuracy, prevalence, null_accuracy, threshold,
                              n_test, n_train, n_positive_test
        per_player      dict  Player ID -> {n, n_positive, roc_auc, f1}
        model_weights   dict  coefficients / importances + feature_names
        feature_names   list of str
        best_params     dict
        task_type       str  ('classification')
        threshold       float  decision threshold used for hard predictions
    """
    if model_type not in VALID_MODELS:
        raise ValueError(
            f"model_type must be one of {sorted(VALID_MODELS)}, got {model_type!r}"
        )
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}"
        )
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    if not covariates:
        raise ValueError("covariates list must not be empty")

    horizon = MODE_HORIZON[mode]

    if verbose:
        print("=" * 72)
        print("Experiment 3: Short-term Load Response")
        if mode == 'prediction':
            print("  Mode: prediction -- morning covariates at t -> Status Decrease at t+1")
        else:
            print("  Mode: causal_framing -- covariates at t (incl. TI Yesterday) -> Status Decrease at t")
        print("=" * 72)
        print(f"  Mode:           {mode}")
        print(f"  Target:         {TARGET}  (target_horizon={horizon})")
        print(f"  Model:          {model_type}")
        print(f"  Lag:            {lag}")
        print(f"  Covariates:     {len(covariates)}")
        for c in covariates:
            print(f"                  . {c}")
        print(f"  Test size:      {test_size}")
        print(f"  Val size:       {val_size}")
        if model_kwargs:
            print(f"  Overrides:      {model_kwargs}")
        print()

    model = _build_model(
        model_type=model_type,
        covariates=covariates,
        lag=lag,
        mode=mode,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        **model_kwargs,
    )
    raw = model.train()

    # Models return probabilities when task_type == 'classification'.
    # Use optimal threshold from model training if available; else 0.5.
    threshold = raw.get('optimal_threshold', 0.5)
    if threshold is None:
        threshold = 0.5

    metrics = _compute_metrics_classification(
        y_true=raw['y_test_true'],
        y_pred_proba=raw['y_test_pred'],
        y_train_true=raw['y_train_true'],
        threshold=threshold,
    )

    per_player: Dict = {}
    meta_test = model.data.get('meta_test')
    if meta_test is not None and not meta_test.empty:
        per_player = _per_player_metrics(
            raw['y_test_true'], raw['y_test_pred'], meta_test, threshold=threshold
        )

    if verbose:
        print(f"  ROC AUC:      {metrics['roc_auc']:.4f}")
        print(f"  Avg Precision:{metrics['avg_precision']:.4f}  "
              f"(prevalence: {metrics['prevalence']:.3f})")
        print(f"  F1:           {metrics['f1']:.4f}  "
              f"(threshold: {metrics['threshold']:.3f})")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  Null Acc:     {metrics['null_accuracy']:.4f}  "
              f"(majority-class baseline)")
        print(f"  N test rows:  {metrics['n_test']}  "
              f"({metrics['n_positive_test']} positives)  |  "
              f"N train rows: {metrics['n_train']}")
        print()

    result = {
        'config': {
            'experiment':     'Experiment3_LoadResponse',
            'mode':           mode,
            'covariates':     covariates,
            'lag':            lag,
            'model_type':     model_type,
            'target':         TARGET,
            'target_horizon': horizon,
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
        'task_type':     raw.get('task_type', 'classification'),
        'threshold':     threshold,
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
    # Prediction-only mode
    results = run_experiment(
        covariates=DEFAULT_COVARIATES,
        lag=3,
        model_type='xgboost',
        mode='prediction',
    )

    print(f"Prediction mode — ROC AUC: {results['metrics']['roc_auc']:.4f}  "
          f"F1: {results['metrics']['f1']:.4f}")

    # Causal-framing mode
    print()
    results_cf = run_experiment(
        covariates=DEFAULT_COVARIATES,
        lag=3,
        model_type='log_reg',
        mode='causal_framing',
    )

    print(f"Causal framing — ROC AUC: {results_cf['metrics']['roc_auc']:.4f}  "
          f"F1: {results_cf['metrics']['f1']:.4f}")
