"""
TabPFN Model for Readiness Prediction

Prior-Data Fitted Networks (TabPFN v2) — a transformer pre-trained on synthetic
tabular datasets that performs in-context learning. No explicit training loop or
hyperparameter search is needed: the model learns from the training set in a single
forward pass at inference time.

Key properties:
- Zero-shot/few-shot learner: fit() stores the training data; predict() runs inference
- Works best on small-to-medium datasets (original sweet spot: < 1000 rows,
  < 100 features), though TabPFN v2 has relaxed these limits considerably
- Supports both classification (TabPFNClassifier) and regression (TabPFNRegressor)
- Returns dict identical to all other models in this package

Install:
    pip install tabpfn

Usage:
    from src.models.tabpfn import TabPFNModel

    model = TabPFNModel(
        target_variable='Status Decrease',
        predictory_columns=[...],
        lag=3,
    )
    results = model.train()
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Optional
import warnings

# src/models/ → src/ (project root / src)
# Adding src/ to path so 'data.data_loader' is importable
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    warnings.warn("TabPFN not installed. Install with: pip install tabpfn")

from data.data_loader import ReadinessDataLoader


class TabPFNModel:
    """
    TabPFN v2 classifier/regressor with optimal threshold selection.

    TabPFN uses in-context learning — no iterative training occurs. The fit() call
    stores the training data; all computation happens during predict_proba()/predict().

    Parameters
    ----------
    target_variable : str
        Name of the target column.
    predictory_columns : list of str
        Names of predictor columns.
    lag : int, default=0
        Number of lag steps for temporal features (passed to data_loader).
    include_previous_target : bool, default=False
        Whether to include the lagged target as a predictor.
    target_horizon : int, default=0
        Number of time steps to shift the target forward.
    test_size : float, default=0.2
        Proportion of dates for the test set.
    val_size : float, default=0.1
        Proportion of dates for the validation set.
    categorical_encoding : str, default='label'
        Encoding strategy for categorical features. TabPFN handles numerics natively;
        label-encoding categoricals is recommended.
    missing_numeric : str, default='mean'
        Imputation strategy for numerical features.
    missing_categorical : str, default='mode'
        Imputation strategy for categorical features.
    standardize : bool, default=False
        Whether to z-score numerical features before passing to TabPFN.
        TabPFN normalises inputs internally, so standardising externally is optional.
    device : str, default='auto'
        Device for TabPFN inference ('cpu', 'cuda', or 'auto').
    n_estimators : int, default=4
        Number of ensembled forward passes (TabPFN v2 parameter).
        Higher values improve stability at the cost of inference time.
    ignore_pretraining_limits : bool, default=False
        If True, bypasses TabPFN v2's built-in sample-size validation that
        raises RuntimeError when running on CPU with more than 1000 rows.
        Set to True when using CPU with large datasets (e.g. RTT.xlsx ~14k rows).
        Has no effect when running on GPU.
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
        categorical_encoding: str = 'label',
        missing_numeric: str = 'mean',
        missing_categorical: str = 'mode',
        standardize: bool = False,
        device: str = 'auto',
        n_estimators: int = 4,
        ignore_pretraining_limits: bool = False,
        random_state: int = 42
    ):
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN is not installed. Install with: pip install tabpfn")

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
        self.device = device
        self.n_estimators = n_estimators
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state

        # Belt-and-suspenders: some TabPFN builds don't honour the
        # ignore_pretraining_limits constructor flag at validation time.
        # Setting the environment variable is the most reliable bypass.
        if ignore_pretraining_limits:
            os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

        self.data_loader = ReadinessDataLoader()
        self.model = None
        self.data = None
        self.task_type = None

    def _detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect classification vs regression."""
        unique_values = y.nunique()
        if unique_values <= 10 and y.dtype in ['int64', 'int32', 'bool']:
            return 'classification'
        # Handle float64 binary (0.0/1.0) resulting from pandas operations
        if unique_values <= 2 and set(y.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return 'classification'
        return 'regression'

    def _load_data(self) -> Dict:
        """Load and preprocess data."""
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
            standardize=self.standardize
        )
        self.task_type = self._detect_task_type(self.data['y_train'])
        return self.data

    def _find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold using Youden's J statistic."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]

    def train(self) -> Dict:
        """
        Complete training pipeline.

        TabPFN performs in-context learning: no iterative optimisation occurs.
        The fit() call simply stores the training data; inference happens at
        predict_proba()/predict() time.

        Returns
        -------
        Dict containing:
            - y_train_true, y_train_pred (probabilities for classification)
            - y_test_true, y_test_pred (probabilities for classification)
            - y_val_true, y_val_pred (probabilities for classification, if val exists)
            - model_weights: {'feature_names': [...]}  (TabPFN has no extractable weights)
            - task_type, best_params
            - metrics: roc_auc, optimal_threshold, f1_optimal, accuracy_optimal,
                      precision_optimal, recall_optimal (for classification)
            - metrics: mse, rmse, r2 (for regression)
        """
        # Load data
        self._load_data()

        # Instantiate TabPFN model
        if self.task_type == 'classification':
            self.model = TabPFNClassifier(
                device=self.device,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
            )
        else:
            self.model = TabPFNRegressor(
                device=self.device,
                n_estimators=self.n_estimators,
                ignore_pretraining_limits=self.ignore_pretraining_limits,
            )

        # TabPFN fit: stores training data (no iterative learning)
        self.model.fit(self.data['X_train'], self.data['y_train'])

        # Generate predictions
        if self.task_type == 'classification':
            y_train_pred = self.model.predict_proba(self.data['X_train'])[:, 1]
            y_test_pred = self.model.predict_proba(self.data['X_test'])[:, 1]
            y_val_pred = (
                self.model.predict_proba(self.data['X_val'])[:, 1]
                if len(self.data['y_val']) > 0 else None
            )
        else:
            y_train_pred = self.model.predict(self.data['X_train'])
            y_test_pred = self.model.predict(self.data['X_test'])
            y_val_pred = (
                self.model.predict(self.data['X_val'])
                if len(self.data['y_val']) > 0 else None
            )

        # TabPFN has no extractable feature importances
        model_weights = {
            'feature_names': self.data['feature_names'],
            'note': 'TabPFN is a black-box transformer; feature importances are not directly available.'
        }

        # Prepare results
        results = {
            'y_train_true': self.data['y_train'].values,
            'y_train_pred': y_train_pred,
            'y_test_true': self.data['y_test'].values,
            'y_test_pred': y_test_pred,
            'model_weights': model_weights,
            'task_type': self.task_type,
            'best_params': {
                'device': self.device,
                'n_estimators': self.n_estimators,
                'ignore_pretraining_limits': self.ignore_pretraining_limits,
            }
        }

        # Add validation predictions if available
        if y_val_pred is not None:
            results['y_val_true'] = self.data['y_val'].values
            results['y_val_pred'] = y_val_pred

        # Compute metrics
        if self.task_type == 'classification':
            # Find optimal threshold on validation set (or train if no val)
            if y_val_pred is not None and len(self.data['y_val']) > 0:
                optimal_threshold = self._find_optimal_threshold(
                    self.data['y_val'].values, y_val_pred
                )
            else:
                optimal_threshold = self._find_optimal_threshold(
                    self.data['y_train'].values, y_train_pred
                )

            y_test_pred_binary = (y_test_pred >= optimal_threshold).astype(int)

            metrics = {
                'roc_auc': roc_auc_score(self.data['y_test'].values, y_test_pred) if len(np.unique(self.data['y_test'].values)) > 1 else None,
                'optimal_threshold': float(optimal_threshold),
                'f1_optimal': f1_score(self.data['y_test'].values, y_test_pred_binary, zero_division=0),
                'accuracy_optimal': accuracy_score(self.data['y_test'].values, y_test_pred_binary),
                'precision_optimal': precision_score(self.data['y_test'].values, y_test_pred_binary, zero_division=0),
                'recall_optimal': recall_score(self.data['y_test'].values, y_test_pred_binary, zero_division=0)
            }
        else:
            metrics = {
                'mse': mean_squared_error(self.data['y_test'].values, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(self.data['y_test'].values, y_test_pred)),
                'r2': r2_score(self.data['y_test'].values, y_test_pred)
            }

        results['metrics'] = metrics

        return results


if __name__ == "__main__":
    predictors = [
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
        'Physical State', 'Sleep Quality (z)', 'Stress (z)',
        'Mood (z)', 'Mental State', 'Overall Wellbeing',
        'Total Distance (ACWR) Yesterday',
        'High Speed Distance (ACWR) Yesterday',
        'Any ACWR Danger', 'Days Since Game', 'Days Until Match',
        'Medical Availability Last 14 Days',
        'Club Attendance Last 14 Days',
        'Position', 'Activity Type Yesterday'
    ]

    model = TabPFNModel(
        target_variable='Status Decrease',
        predictory_columns=predictors,
        lag=3,
        include_previous_target=True,
        test_size=0.2,
        val_size=0.1,
        device='auto',
        n_estimators=4,
    )

    results = model.train()
    print(f"Task: {results['task_type']}")
    print(f"Best params: {results['best_params']}")
    print(f"Metrics: {results['metrics']}")
