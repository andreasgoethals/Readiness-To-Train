"""
src/models — ML model classes for the Readiness-to-Train project.

All models share the same .train() interface and use ReadinessDataLoader internally.
"""

from .log_reg import LogisticRegressionModel
from .xgboost import XGBoostModel
from .catboost import CatBoostModel
from .tabpfn import TabPFNModel

__all__ = [
    'LogisticRegressionModel',
    'XGBoostModel',
    'CatBoostModel',
    'TabPFNModel',
]
