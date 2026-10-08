"""
Antigravity ML Assignment Package - Polynomial Regression
Roll No: BT2024004
"""

from .data_loader import load_dataset
from .models import PolynomialRegressionModel
from .evaluate import evaluate_predictions, cross_validate_model, compute_information_criteria

__all__ = [
    'load_dataset',
    'PolynomialRegressionModel',
    'evaluate_predictions',
    'cross_validate_model',
    'compute_information_criteria',
]
