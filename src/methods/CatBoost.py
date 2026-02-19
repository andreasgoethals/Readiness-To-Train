"""
CatBoost Model for Readiness Prediction

Streamlined implementation with:
- Automatic task type detection (classification vs regression)
- Native categorical feature handling via cat_features parameter
- Hyperparameter optimization using Optuna
- Optimal threshold selection using Youden's J statistic
- Returns dict only, no file I/O or print statements
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
from typing import List, Dict, Optional
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)
import optuna

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not installed. Install with: pip install catboost")

from data.data_loader import ReadinessDataLoader


class CatBoostModel:
    """CatBoost classifier/regressor with HPO and optimal threshold selection."""

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
        hpo_trials: int = 50,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")

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
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.data_loader = ReadinessDataLoader()
        self.model = None
        self.data = None
        self.task_type = None
        self.best_params = None
        self.cat_features_indices = None

        # CatBoost writes training logs to a 'catboost_info' directory by default.
        # Redirect this to a temporary directory to prevent polluting the project root.
        self._catboost_train_dir = tempfile.mkdtemp(prefix='catboost_')

    def _detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect classification vs regression."""
        unique_values = y.nunique()
        if unique_values <= 10 and y.dtype in ['int64', 'int32', 'bool']:
            return 'classification'
        else:
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

        # Get categorical feature indices for CatBoost
        categorical_features = self.data['categorical_features']
        feature_names = self.data['feature_names']

        # Find indices of categorical features
        self.cat_features_indices = [
            i for i, name in enumerate(feature_names)
            if name in categorical_features
        ]

        return self.data

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        iterations = trial.suggest_int('iterations', 100, 1000)
        depth = trial.suggest_int('depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
        random_strength = trial.suggest_float('random_strength', 0.0, 10.0)

        # Create model based on task type
        if self.task_type == 'classification':
            model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                random_seed=self.random_state,
                verbose=0,
                auto_class_weights='Balanced',
                cat_features=self.cat_features_indices,
                train_dir=self._catboost_train_dir
            )
        else:
            model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                random_seed=self.random_state,
                verbose=0,
                cat_features=self.cat_features_indices,
                train_dir=self._catboost_train_dir
            )

        # Train on training set with early stopping on validation set
        if len(self.data['y_val']) > 0:
            model.fit(
                self.data['X_train'],
                self.data['y_train'],
                eval_set=(self.data['X_val'], self.data['y_val']),
                early_stopping_rounds=50,
                verbose=False
            )

            # Evaluate on validation set
            if self.task_type == 'classification':
                y_val_pred = model.predict(self.data['X_val'])
                score = f1_score(self.data['y_val'], y_val_pred, zero_division=0)
            else:
                y_val_pred = model.predict(self.data['X_val'])
                score = -mean_squared_error(self.data['y_val'], y_val_pred)
        else:
            # No validation set - train without early stopping
            model.fit(
                self.data['X_train'],
                self.data['y_train'],
                verbose=False
            )

            # Use training score (not ideal)
            if self.task_type == 'classification':
                y_train_pred = model.predict(self.data['X_train'])
                score = f1_score(self.data['y_train'], y_train_pred, zero_division=0)
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
            - model_weights (feature importances dict)
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
            iterations = self.best_params['iterations']
            depth = self.best_params['depth']
            learning_rate = self.best_params['learning_rate']
            l2_leaf_reg = self.best_params['l2_leaf_reg']
            random_strength = self.best_params['random_strength']
        else:
            iterations = self.iterations
            depth = self.depth
            learning_rate = self.learning_rate
            l2_leaf_reg = 3.0
            random_strength = 1.0

        # Train final model with best parameters
        if self.task_type == 'classification':
            self.model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                random_seed=self.random_state,
                verbose=0,
                auto_class_weights='Balanced',
                cat_features=self.cat_features_indices,
                train_dir=self._catboost_train_dir
            )
        else:
            self.model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                random_seed=self.random_state,
                verbose=0,
                cat_features=self.cat_features_indices,
                train_dir=self._catboost_train_dir
            )

        # Train with early stopping if validation set available
        if len(self.data['y_val']) > 0:
            self.model.fit(
                self.data['X_train'],
                self.data['y_train'],
                eval_set=(self.data['X_val'], self.data['y_val']),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(
                self.data['X_train'],
                self.data['y_train'],
                verbose=False
            )

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
        feature_importances = self.model.get_feature_importance()

        # Get best_iteration safely
        best_iteration = None
        if hasattr(self.model, 'best_iteration_') and self.model.best_iteration_ is not None:
            best_iteration = int(self.model.best_iteration_)

        model_weights = {
            'feature_importances': feature_importances.tolist(),
            'feature_names': self.data['feature_names'],
            'best_iteration': best_iteration,
            'categorical_feature_indices': self.cat_features_indices
        }

        # Prepare results
        results = {
            'y_train_true': self.data['y_train'].values,
            'y_train_pred': y_train_pred,
            'y_test_true': self.data['y_test'].values,
            'y_test_pred': y_test_pred,
            'model_weights': model_weights,
            'task_type': self.task_type,
            'best_params': self.best_params if self.hpo_trials > 0 else {
                'iterations': iterations,
                'depth': depth,
                'learning_rate': learning_rate
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
    predictors = [
        'Fatigue (z)', 'Readiness (z)', 'Soreness (z)',
        'Physical State', 'Sleep Quality (z)', 'Stress (z)',
        'Mood (z)', 'Mental State', 'Overall Wellbeing',
        'Total Distance (ACWR) Yesterday',
        'High Speed Distance (ACWR) Yesterday',
        'Any ACWR Danger', 'Days Since Game',
        'Medical Availability Last 14 Days',
        'Club Attendance Last 14 Days',
        'Position', 'Activity Type Yesterday',
        'Comment Category Yesterday'
    ]

    model = CatBoostModel(
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
