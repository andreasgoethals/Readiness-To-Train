"""
XGBoost Model for Readiness Prediction

Streamlined implementation with:
- Automatic task type detection (classification vs regression)
- Optimal threshold selection using Youden's J statistic
- Returns dict only, no file I/O or print statements
- No HPO (as requested by user)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Optional

# src/models/ → src/ (project root / src)
# Adding src/ to path so 'data.data_loader' is importable
sys.path.append(str(Path(__file__).parent.parent))

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)

from data.data_loader import ReadinessDataLoader


class XGBoostModel:
    """XGBoost classifier/regressor with optimal threshold selection."""

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
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 10,
        random_state: int = 42
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
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

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
        """Find optimal threshold using Youden's J statistic on validation set."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

    def train(self) -> Dict:
        """
        Complete training pipeline.

        Returns
        -------
        Dict containing:
            - y_train_true, y_train_pred (probabilities for classification)
            - y_test_true, y_test_pred (probabilities for classification)
            - y_val_true, y_val_pred (probabilities for classification, if val exists)
            - model_weights (feature importances dict)
            - task_type, best_params
            - metrics: roc_auc, optimal_threshold, f1_optimal, accuracy_optimal,
                      precision_optimal, recall_optimal (for classification)
            - metrics: mse, rmse, r2 (for regression)
        """
        # Load data
        self._load_data()

        # Auto-compute scale_pos_weight for imbalanced classification
        if self.task_type == 'classification' and self.scale_pos_weight is None:
            n_neg = (self.data['y_train'] == 0).sum()
            n_pos = (self.data['y_train'] == 1).sum()
            if n_pos > 0:
                self.scale_pos_weight = n_neg / n_pos

        # Create XGBoost parameters
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': -1
        }

        # Add task-specific parameters
        if self.task_type == 'classification':
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
            if self.scale_pos_weight is not None:
                params['scale_pos_weight'] = self.scale_pos_weight
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'

        # Create XGBoost model
        if self.task_type == 'classification':
            self.model = xgb.XGBClassifier(**params, n_estimators=self.n_estimators)
        else:
            self.model = xgb.XGBRegressor(**params, n_estimators=self.n_estimators)

        # Train with early stopping if validation set available
        if len(self.data['y_val']) > 0:
            self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            self.model.fit(
                self.data['X_train'],
                self.data['y_train'],
                eval_set=[(self.data['X_val'], self.data['y_val'])],
                verbose=False
            )
        else:
            self.model.fit(self.data['X_train'], self.data['y_train'])

        # Generate predictions
        if self.task_type == 'classification':
            y_train_pred = self.model.predict_proba(self.data['X_train'])[:, 1]
            y_test_pred = self.model.predict_proba(self.data['X_test'])[:, 1]
            y_val_pred = self.model.predict_proba(self.data['X_val'])[:, 1] if len(self.data['y_val']) > 0 else None
        else:
            y_train_pred = self.model.predict(self.data['X_train'])
            y_test_pred = self.model.predict(self.data['X_test'])
            y_val_pred = self.model.predict(self.data['X_val']) if len(self.data['y_val']) > 0 else None

        # Extract feature importances
        feature_importances = self.model.feature_importances_

        # Get best_iteration safely
        best_iteration = None
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            best_iteration = int(self.model.best_iteration)

        model_weights = {
            'feature_importances': feature_importances.tolist(),
            'feature_names': self.data['feature_names'],
            'best_iteration': best_iteration
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
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree
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

            # Apply optimal threshold to test predictions
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
            # Regression metrics
            metrics = {
                'mse': mean_squared_error(self.data['y_test'].values, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(self.data['y_test'].values, y_test_pred)),
                'r2': r2_score(self.data['y_test'].values, y_test_pred)
            }

        results['metrics'] = metrics

        return results


if __name__ == "__main__":
    # Example usage
    # NOTE: Activity Type Today is determined AFTER the morning assessment
    # (it is derived from the next row's Activity Type Yesterday). It should
    # NOT be included as a predictor since it is not available at prediction time.
    # Only variables known at the time of the morning assessment are valid predictors.
    predictors = [
        # Morning wellness assessment (covariates)
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
        'Physical State', 'Sleep Quality (z)', 'Stress (z)',
        'Mood (z)', 'Mental State', 'Overall Wellbeing',
        # Yesterday's workload data (fully known)
        'Total Distance (ACWR) Yesterday',
        'High Speed Distance (ACWR) Yesterday',
        'Any ACWR Danger',
        # Temporal features (known at morning assessment)
        'Days Since Game', 'Days Until Match',
        # Historical context
        'Medical Availability Last 14 Days',
        'Club Attendance Last 14 Days',
        # Player profile
        'Position',
        # Previous day's activity (known — it already happened)
        'Activity Type Yesterday'
    ]

    model = XGBoostModel(
        target_variable='Status Decrease',
        predictory_columns=predictors,
        lag=2,
        include_previous_target=True,
        test_size=0.2,
        val_size=0.1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    results = model.train()
    print(f"Task: {results['task_type']}")
    print(f"Best params: {results['best_params']}")
    print(f"Metrics: {results['metrics']}")
