"""
Experiment 1: Compare All Models on a Readiness Prediction Task
================================================================

This script trains and evaluates all available models (LogReg, XGBoost, CatBoost)
on a configurable prediction task. It is designed to be called programmatically
from notebooks or other scripts, not just run standalone.

PREDICTOR SELECTION: Exclusion-based — you specify which columns to EXCLUDE,
and all remaining valid columns become predictors. Metadata columns (Date,
Playerkey, Player ID, Status, and the target variable itself) are always
excluded automatically.

TARGET ALIGNMENT: The target variable is always shifted forward by 1 time step
(target_horizon=1) so that features from time t predict the outcome at t+1.

Usage:
    # From a notebook or another script:
    from scripts.Experiment1 import main
    results = main(
        exclude_columns=['Comment Yesterday', 'Comment Category Yesterday'],
        target='Status Decrease',
        lag=3
    )
"""

import sys
import traceback
from pathlib import Path
from typing import List, Dict

# Add src to path so model imports work
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import pandas as pd

from methods.LogReg import LogisticRegressionModel
from methods.XGBoost import XGBoostModel
from methods.CatBoost import CatBoostModel
from data.data_loader import ReadinessDataLoader


# =============================================================================
# COLUMNS THAT ARE ALWAYS EXCLUDED (metadata, identifiers, raw target)
# These should never be used as predictors regardless of user configuration.
# =============================================================================

ALWAYS_EXCLUDED = {
    'Date', 'Playerkey', 'Player ID', 'Status'
}


# =============================================================================
# HELPER: RESOLVE PREDICTORS FROM EXCLUSIONS
# =============================================================================

def _resolve_predictors(exclude_columns: List[str], target: str) -> List[str]:
    """
    Determine predictor columns by loading the processed data and subtracting
    the excluded columns, always-excluded metadata columns, and the target.

    Parameters
    ----------
    exclude_columns : List[str]
        User-specified columns to exclude from predictors.
    target : str
        Target variable name (also excluded from predictors).

    Returns
    -------
    List[str]
        Sorted list of predictor column names.
    """
    # Load processed data to get all available columns
    loader = ReadinessDataLoader()
    df = pd.read_csv(loader.data_path, nrows=1)
    all_columns = set(df.columns.tolist())

    # Build the full exclusion set
    exclusion_set = ALWAYS_EXCLUDED | set(exclude_columns) | {target}

    # Validate that excluded columns actually exist
    invalid_exclusions = set(exclude_columns) - all_columns
    if invalid_exclusions:
        print(f"  Warning: these exclude_columns are not in the data and will be ignored: {invalid_exclusions}")

    # Predictors = all columns minus exclusions
    predictors = sorted(all_columns - exclusion_set)

    return predictors


# =============================================================================
# INDIVIDUAL MODEL RUNNERS
# =============================================================================

def run_logreg(predictors: List[str], target: str, lag: int,
               target_horizon: int = 1,
               test_size: float = 0.2, val_size: float = 0.1,
               hpo_trials: int = 20, random_state: int = 42) -> Dict:
    """
    Train and evaluate a Logistic Regression model.

    Uses one-hot encoding (appropriate for linear models) and standardization
    (required so that regularization treats all features equally).
    """
    model = LogisticRegressionModel(
        target_variable=target,
        predictory_columns=predictors,
        lag=lag,
        include_previous_target=True,
        target_horizon=target_horizon,
        test_size=test_size,
        val_size=val_size,
        hpo_trials=hpo_trials,
        categorical_encoding='one-hot',
        standardize=True,
        random_state=random_state
    )
    return model.train()


def run_xgboost(predictors: List[str], target: str, lag: int,
                target_horizon: int = 1,
                test_size: float = 0.2, val_size: float = 0.1,
                random_state: int = 42) -> Dict:
    """
    Train and evaluate an XGBoost model.

    Uses label encoding (efficient for tree-based models) and no
    standardization (trees are scale-invariant).
    """
    model = XGBoostModel(
        target_variable=target,
        predictory_columns=predictors,
        lag=lag,
        include_previous_target=True,
        target_horizon=target_horizon,
        test_size=test_size,
        val_size=val_size,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        categorical_encoding='label',
        standardize=False,
        random_state=random_state
    )
    return model.train()


def run_catboost(predictors: List[str], target: str, lag: int,
                 target_horizon: int = 1,
                 test_size: float = 0.2, val_size: float = 0.1,
                 hpo_trials: int = 20, random_state: int = 42) -> Dict:
    """
    Train and evaluate a CatBoost model.

    Uses label encoding (CatBoost handles categorical features natively)
    and no standardization (trees are scale-invariant).
    """
    model = CatBoostModel(
        target_variable=target,
        predictory_columns=predictors,
        lag=lag,
        include_previous_target=True,
        target_horizon=target_horizon,
        test_size=test_size,
        val_size=val_size,
        hpo_trials=hpo_trials,
        iterations=500,
        depth=6,
        learning_rate=0.1,
        categorical_encoding='label',
        standardize=False,
        random_state=random_state
    )
    return model.train()


# =============================================================================
# MAIN EXPERIMENT FUNCTION
# =============================================================================

def main(
    exclude_columns: List[str],
    target: str,
    lag: int,
    target_horizon: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    hpo_trials: int = 20,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Run Experiment 1: train all three models and compare performance.

    This is the main entry point for the experiment. It resolves predictors
    from the exclusion list, trains LogReg, XGBoost, and CatBoost on the
    same data split, collects their results, and returns a structured
    dictionary for downstream analysis and visualization.

    Parameters
    ----------
    exclude_columns : List[str]
        Column names to EXCLUDE from predictors. All other valid columns
        (minus metadata and the target) will be used as predictors.
    target : str
        Target variable name (e.g., 'Status Decrease').
    lag : int
        Number of lag steps for temporal features (e.g., 3 creates t-1, t-2, t-3).
    target_horizon : int, default=1
        Number of time steps to shift the target forward. With target_horizon=1,
        features from time t predict the target at time t+1.
    test_size : float, default=0.2
        Proportion of dates for the test set.
    val_size : float, default=0.1
        Proportion of dates for the validation set.
    hpo_trials : int, default=20
        Number of Optuna HPO trials for LogReg and CatBoost.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress and results to stdout.

    Returns
    -------
    Dict with keys:
        - 'config': dict of experiment configuration parameters
        - 'models': dict mapping model_name -> results dict (or None if failed)
        - 'comparison': list of dicts with model name and metrics (successful only)
        - 'summary': dict with total, successful, failed counts
    """
    # Resolve predictors from exclusion list
    predictors = _resolve_predictors(exclude_columns, target)

    # Store experiment configuration
    config = {
        'predictors': predictors,
        'exclude_columns': exclude_columns,
        'target': target,
        'lag': lag,
        'target_horizon': target_horizon,
        'test_size': test_size,
        'val_size': val_size,
        'hpo_trials': hpo_trials,
        'random_state': random_state
    }

    if verbose:
        print("=" * 80)
        print("EXPERIMENT 1: Comparing All Models")
        print("=" * 80)
        print(f"\n  Target Variable:    {target}")
        print(f"  Target Horizon:     t+{target_horizon}")
        print(f"  Excluded Columns:   {exclude_columns}")
        print(f"  Number of Predictors: {len(predictors)}")
        print(f"  Predictors:         {predictors}")
        print(f"  Lag:                  {lag}")
        print(f"  Test Size:            {test_size}")
        print(f"  Validation Size:      {val_size}")
        print(f"  HPO Trials:           {hpo_trials}")

    # Define all models to run
    model_runners = {
        'Logistic Regression': lambda: run_logreg(
            predictors, target, lag, target_horizon,
            test_size, val_size, hpo_trials, random_state
        ),
        'XGBoost': lambda: run_xgboost(
            predictors, target, lag, target_horizon,
            test_size, val_size, random_state
        ),
        'CatBoost': lambda: run_catboost(
            predictors, target, lag, target_horizon,
            test_size, val_size, hpo_trials, random_state
        ),
    }

    # Run each model with error handling
    model_results = {}
    for model_name, runner in model_runners.items():
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  Training {model_name}...")
            print(f"{'─' * 60}")

        try:
            results = runner()
            model_results[model_name] = results

            if verbose:
                _print_metrics(results, model_name)

        except Exception as e:
            model_results[model_name] = None
            if verbose:
                print(f"\n  {model_name} FAILED: {type(e).__name__}: {str(e)}")
                traceback.print_exc()

    # Build comparison table for successful models
    comparison = []
    for model_name, results in model_results.items():
        if results is not None and results['task_type'] == 'classification':
            metrics = results['metrics']
            comparison.append({
                'model': model_name,
                'roc_auc': metrics.get('roc_auc'),
                'f1': metrics.get('f1_optimal'),
                'accuracy': metrics.get('accuracy_optimal'),
                'precision': metrics.get('precision_optimal'),
                'recall': metrics.get('recall_optimal'),
                'optimal_threshold': metrics.get('optimal_threshold'),
                'best_params': results.get('best_params')
            })

    # Summary counts
    n_success = sum(1 for r in model_results.values() if r is not None)
    n_failed = sum(1 for r in model_results.values() if r is None)

    summary = {
        'total': len(model_runners),
        'successful': n_success,
        'failed': n_failed,
        'failed_models': [name for name, r in model_results.items() if r is None]
    }

    if verbose:
        _print_summary(comparison, summary)

    return {
        'config': config,
        'models': model_results,
        'comparison': comparison,
        'summary': summary
    }


# =============================================================================
# PRINTING HELPERS (for verbose mode)
# =============================================================================

def _print_metrics(results: Dict, model_name: str):
    """Print formatted metrics for a single model."""
    print(f"\n  {model_name} — {results['task_type']}")
    print(f"  Best Params: {results['best_params']}")

    metrics = results['metrics']
    if results['task_type'] == 'classification':
        roc = f"{metrics['roc_auc']:.4f}" if metrics.get('roc_auc') is not None else "N/A"
        print(f"    ROC AUC:         {roc}")
        print(f"    Optimal Thresh:  {metrics['optimal_threshold']:.4f}")
        print(f"    F1 (optimal):    {metrics['f1_optimal']:.4f}")
        print(f"    Accuracy:        {metrics['accuracy_optimal']:.4f}")
        print(f"    Precision:       {metrics['precision_optimal']:.4f}")
        print(f"    Recall:          {metrics['recall_optimal']:.4f}")
    else:
        print(f"    MSE:             {metrics['mse']:.4f}")
        print(f"    RMSE:            {metrics['rmse']:.4f}")
        print(f"    R²:              {metrics['r2']:.4f}")


def _print_summary(comparison: list, summary: dict):
    """Print experiment summary and comparison table."""
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  Total: {summary['total']}  |  "
          f"Successful: {summary['successful']}  |  "
          f"Failed: {summary['failed']}")

    if summary['failed_models']:
        print(f"  Failed: {', '.join(summary['failed_models'])}")

    if comparison:
        print(f"\n  {'Model':<25} {'ROC AUC':<10} {'F1':<10} {'Accuracy':<10} "
              f"{'Precision':<10} {'Recall':<10}")
        print(f"  {'─' * 75}")
        for row in comparison:
            roc = f"{row['roc_auc']:.4f}" if row['roc_auc'] is not None else "N/A"
            print(f"  {row['model']:<25} {roc:<10} {row['f1']:.4f}     "
                  f"{row['accuracy']:.4f}     {row['precision']:.4f}     "
                  f"{row['recall']:.4f}")

    print(f"\n{'=' * 80}\n")


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example: exclude free-text comments and run with lag=3
    results = main(
        exclude_columns=['Comment Yesterday'],
        target='Status Decrease',
        lag=3
    )
