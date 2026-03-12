"""
Linear Regression Model for Readiness Prediction

Streamlined implementation with:
- Ridge regression (L2 regularisation) for continuous targets
- RidgeClassifier as fallback for binary/multi-class targets
- Hyperparameter optimisation of alpha via Optuna
- Automatic task-type detection (classification vs regression)
- Returns dict only, no file I/O or print statements
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Optional

# src/models/ → src/  (project root / src)
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score,
)
import optuna

from data.data_loader import ReadinessDataLoader


class LinearRegressionModel:
    """
    Ridge regression / RidgeClassifier with Optuna HPO on regularisation strength.

    For regression targets (default use-case): fits Ridge(alpha) where alpha is
    optimised on the validation set via MSE minimisation.

    For binary/multi-class targets (auto-detected): falls back to RidgeClassifier
    with Youden-J optimal threshold selection on the validation set.

    Parameters
    ----------
    target_variable : str
        Name of the target column.
    predictory_columns : list of str
        Predictor columns.
    lag : int, default=0
        Number of lag steps for temporal features (player-grouped in data_loader).
    include_previous_target : bool, default=False
        Include lagged target as a predictor.
    target_horizon : int, default=0
        Number of time steps to shift the target forward within each player.
    test_size : float, default=0.2
        Fraction of dates for the test split (temporal: latest dates).
    val_size : float, default=0.1
        Fraction of dates for the validation split.
    categorical_encoding : str, default='one-hot'
        Feature encoding strategy ('one-hot' or 'label').
        One-hot is recommended for linear models (ensures correct scale).
    missing_numeric : str, default='mean'
        Imputation strategy for numerical features.
    missing_categorical : str, default='mode'
        Imputation strategy for categorical features.
    standardize : bool, default=True
        Z-score normalise numerical features. Should be True for Ridge since
        the regularisation penalty is scale-sensitive.
    hpo_trials : int, default=50
        Number of Optuna trials for alpha HPO. Set 0 to skip HPO.
    alpha : float, default=1.0
        Regularisation strength used when hpo_trials=0.
    max_iter : int, default=1000
        Maximum iterations for the Ridge solver.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
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
        standardize: bool = True,
        hpo_trials: int = 50,
        alpha: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.target_variable = target_variable
        self.predictory_columns = predictory_columns
        self.lag = lag
        self.include_previous_target = include_previous_target
        self.target_horizon = target_horizon
        self.test_size = test_size
        self.val_size = val_size
        self.categorical_encoding = categorical_encoding
        self.missing_numeric = missing_numeric
        self.missing_categorical = missing_categorical
        self.standardize = standardize
        self.hpo_trials = hpo_trials
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

        self.data_loader = ReadinessDataLoader()
        self.model = None
        self.data = None
        self.task_type = None
        self.best_params = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect classification vs regression."""
        unique_values = y.nunique()
        if unique_values <= 10 and y.dtype in ['int64', 'int32', 'bool']:
            return 'classification'
        if unique_values <= 2 and set(y.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return 'classification'
        return 'regression'

    def _load_data(self) -> Dict:
        """Load and preprocess data via the shared data loader."""
        self.data = self.data_loader.create_dataset(
            target_variable=self.target_variable,
            predictory_columns=self.predictory_columns,
            lag=self.lag,
            include_previous_target=self.include_previous_target,
            target_horizon=self.target_horizon,
            test_size=self.test_size,
            val_size=self.val_size,
            categorical_encoding=self.categorical_encoding,
            missing_numeric=self.missing_numeric,
            missing_categorical=self.missing_categorical,
            standardize=self.standardize,
        )
        self.task_type = self._detect_task_type(self.data['y_train'])
        return self.data

    def _build_model(self, alpha: float):
        """Instantiate Ridge or RidgeClassifier with the given alpha."""
        if self.task_type == 'regression':
            return Ridge(alpha=alpha, max_iter=self.max_iter)
        else:
            return RidgeClassifier(alpha=alpha, max_iter=self.max_iter)

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: optimise alpha on the validation set."""
        alpha = trial.suggest_float('alpha', 1e-4, 1e3, log=True)
        model = self._build_model(alpha)
        model.fit(self.data['X_train'], self.data['y_train'])

        eval_y = self.data['y_val'] if len(self.data['y_val']) > 0 else self.data['y_train']
        eval_X = self.data['X_val'] if len(self.data['y_val']) > 0 else self.data['X_train']

        if self.task_type == 'regression':
            y_pred = model.predict(eval_X)
            return -mean_squared_error(eval_y, y_pred)   # maximise → minimise MSE
        else:
            # RidgeClassifier does not expose predict_proba; use decision_function
            scores = model.decision_function(eval_X)
            if len(np.unique(eval_y)) > 1:
                return roc_auc_score(eval_y, scores)
            return 0.0

    def _run_hpo(self) -> Dict:
        """Run Optuna HPO for alpha."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(self._objective, n_trials=self.hpo_trials, show_progress_bar=False)
        self.best_params = study.best_params
        return self.best_params

    def _find_optimal_threshold(self, y_true, scores):
        """Find optimal threshold using Youden's J on decision-function scores."""
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        j_scores = tpr - fpr
        return float(thresholds[np.argmax(j_scores)])

    # ── public API ────────────────────────────────────────────────────────────

    def train(self) -> Dict:
        """
        Complete training pipeline.

        Returns
        -------
        Dict containing:
            y_train_true, y_train_pred
            y_test_true,  y_test_pred
            y_val_true,   y_val_pred   (only when val set is non-empty)
            model_weights : {'coefficients', 'intercept', 'feature_names'}
            task_type, best_params
            metrics (regression): mse, rmse, r2
            metrics (classification): roc_auc, optimal_threshold, f1_optimal,
                                      accuracy_optimal, precision_optimal, recall_optimal
        """
        self._load_data()

        # ── HPO or use default alpha ──
        if self.hpo_trials > 0:
            self._run_hpo()
            alpha = self.best_params['alpha']
        else:
            alpha = self.alpha
            self.best_params = {'alpha': alpha}

        # ── Train final model ──
        self.model = self._build_model(alpha)
        self.model.fit(self.data['X_train'], self.data['y_train'])

        # ── Predictions ──
        if self.task_type == 'regression':
            y_train_pred = self.model.predict(self.data['X_train'])
            y_test_pred  = self.model.predict(self.data['X_test'])
            y_val_pred   = self.model.predict(self.data['X_val']) if len(self.data['y_val']) > 0 else None
        else:
            # Classification: expose decision_function scores (analogous to probabilities)
            y_train_pred = self.model.decision_function(self.data['X_train'])
            y_test_pred  = self.model.decision_function(self.data['X_test'])
            y_val_pred   = (self.model.decision_function(self.data['X_val'])
                           if len(self.data['y_val']) > 0 else None)

        # ── Model weights ──
        coef = self.model.coef_
        model_weights = {
            'coefficients': (coef.tolist() if coef.ndim == 1
                             else coef[0].tolist()),    # 2-D for RidgeClassifier
            'intercept': float(
                self.model.intercept_[0]
                if isinstance(self.model.intercept_, np.ndarray)
                else self.model.intercept_
            ),
            'feature_names': self.data['feature_names'],
        }

        # ── Build results dict ──
        results = {
            'y_train_true': self.data['y_train'].values,
            'y_train_pred': y_train_pred,
            'y_test_true':  self.data['y_test'].values,
            'y_test_pred':  y_test_pred,
            'model_weights': model_weights,
            'task_type':    self.task_type,
            'best_params':  self.best_params,
        }
        if y_val_pred is not None:
            results['y_val_true'] = self.data['y_val'].values
            results['y_val_pred'] = y_val_pred

        # ── Metrics ──
        if self.task_type == 'regression':
            metrics = {
                'mse':  float(mean_squared_error(self.data['y_test'].values, y_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(self.data['y_test'].values, y_test_pred))),
                'r2':   float(r2_score(self.data['y_test'].values, y_test_pred)),
            }
        else:
            # Threshold on val (or train if no val)
            if y_val_pred is not None:
                thresh = self._find_optimal_threshold(self.data['y_val'].values, y_val_pred)
            else:
                thresh = self._find_optimal_threshold(self.data['y_train'].values, y_train_pred)

            y_test_bin = (y_test_pred >= thresh).astype(int)
            metrics = {
                'roc_auc': (roc_auc_score(self.data['y_test'].values, y_test_pred)
                            if len(np.unique(self.data['y_test'].values)) > 1 else None),
                'optimal_threshold':  float(thresh),
                'f1_optimal':         f1_score(self.data['y_test'].values, y_test_bin, zero_division=0),
                'accuracy_optimal':   accuracy_score(self.data['y_test'].values, y_test_bin),
                'precision_optimal':  precision_score(self.data['y_test'].values, y_test_bin, zero_division=0),
                'recall_optimal':     recall_score(self.data['y_test'].values, y_test_bin, zero_division=0),
            }

        results['metrics'] = metrics
        return results


if __name__ == '__main__':
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
        'Position', 'Activity Type Yesterday',
        'Training Intensity Yesterday',
    ]

    model = LinearRegressionModel(
        target_variable='Match Intensity Yesterday',
        predictory_columns=predictors,
        lag=1,
        target_horizon=1,
        test_size=0.2,
        val_size=0.1,
        standardize=True,
        hpo_trials=50,
    )

    results = model.train()
    print(f"Task:        {results['task_type']}")
    print(f"Best params: {results['best_params']}")
    print(f"Metrics:     {results['metrics']}")
