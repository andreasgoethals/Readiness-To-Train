"""
Logistic Regression Model for Readiness Prediction

Streamlined implementation with:
- Automatic task type detection (classification vs regression)
- Hyperparameter optimization using Optuna
- Optimal threshold selection using Youden's J statistic
- Returns dict only, no file I/O or print statements
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Optional

# src/models/ → src/ (project root / src)
# Adding src/ to path so 'data.data_loader' is importable
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)
import optuna

from data.data_loader import ReadinessDataLoader


class LogisticRegressionModel:
    """Logistic Regression classifier/regressor with HPO and optimal threshold selection."""

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
        C: float = 1.0,
        penalty: str = 'l2',
        max_iter: int = 1000,
        class_weight: Optional[str] = 'balanced',
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
        self.hpo_trials = hpo_trials
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.data_loader = ReadinessDataLoader()
        self.model = None
        self.data = None
        self.task_type = None
        self.best_params = None

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

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        C = trial.suggest_float('C', 0.001, 100.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'

        if self.task_type == 'classification':
            model = LogisticRegression(
                C=C,
                penalty=penalty,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver=solver,
                verbose=0
            )
        else:
            alpha = 1.0 / C
            model = Ridge(
                alpha=alpha,
                max_iter=self.max_iter,
                random_state=self.random_state
            )

        model.fit(self.data['X_train'], self.data['y_train'])

        if len(self.data['y_val']) > 0:
            if self.task_type == 'classification':
                # Use ROC AUC with predicted probabilities — F1 at 0.5 threshold
                # is unreliable with severe class imbalance (~3-5% positive)
                y_val_proba = model.predict_proba(self.data['X_val'])[:, 1]
                if len(np.unique(self.data['y_val'])) > 1:
                    score = roc_auc_score(self.data['y_val'], y_val_proba)
                else:
                    score = 0.0
            else:
                y_val_pred = model.predict(self.data['X_val'])
                score = -mean_squared_error(self.data['y_val'], y_val_pred)
        else:
            if self.task_type == 'classification':
                y_train_proba = model.predict_proba(self.data['X_train'])[:, 1]
                if len(np.unique(self.data['y_train'])) > 1:
                    score = roc_auc_score(self.data['y_train'], y_train_proba)
                else:
                    score = 0.0
            else:
                y_train_pred = model.predict(self.data['X_train'])
                score = -mean_squared_error(self.data['y_train'], y_train_pred)

        return score

    def _run_hpo(self) -> Dict:
        """Run hyperparameter optimization."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(self._objective, n_trials=self.hpo_trials, show_progress_bar=False)
        self.best_params = study.best_params
        return self.best_params

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
            - model_weights (coefficients dict)
            - task_type, best_params
            - metrics: roc_auc, optimal_threshold, f1_optimal, accuracy_optimal,
                      precision_optimal, recall_optimal (for classification)
            - metrics: mse, rmse, r2 (for regression)
        """
        # Load data
        self._load_data()

        # Run HPO
        if self.hpo_trials > 0:
            self._run_hpo()
            C = self.best_params['C']
            penalty = self.best_params.get('penalty', 'l2')
        else:
            C = self.C
            penalty = self.penalty

        # Train final model
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'

        if self.task_type == 'classification':
            self.model = LogisticRegression(
                C=C,
                penalty=penalty,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver=solver,
                verbose=0
            )
        else:
            alpha = 1.0 / C
            self.model = Ridge(
                alpha=alpha,
                max_iter=self.max_iter,
                random_state=self.random_state
            )

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

        # Extract model weights
        model_weights = {
            'coefficients': self.model.coef_.tolist() if hasattr(self.model, 'coef_') else None,
            'intercept': float(self.model.intercept_[0] if isinstance(self.model.intercept_, np.ndarray) else self.model.intercept_),
            'feature_names': self.data['feature_names']
        }

        # Prepare results
        results = {
            'y_train_true': self.data['y_train'].values,
            'y_train_pred': y_train_pred,
            'y_test_true': self.data['y_test'].values,
            'y_test_pred': y_test_pred,
            'model_weights': model_weights,
            'task_type': self.task_type,
            'best_params': self.best_params if self.hpo_trials > 0 else {'C': C, 'penalty': penalty}
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

    model = LogisticRegressionModel(
        target_variable='Status Decrease',
        predictory_columns=predictors,
        lag=3,
        include_previous_target=True,
        test_size=0.2,
        val_size=0.1,
        hpo_trials=50
    )

    results = model.train()
    print(f"Task: {results['task_type']}")
    print(f"Best params: {results['best_params']}")
    print(f"Metrics: {results['metrics']}")
