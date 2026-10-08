"""
Evaluation metrics and validation utilities.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate primary evaluation metrics: MSE and R2 score.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    residuals = y_true - y_pred
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'residual_mean': float(np.mean(residuals)),
        'residual_std': float(np.std(residuals)),
    }


def cross_validate_model(model_cls, X: np.ndarray, y: np.ndarray, n_splits: int = 10, random_state: int = 42, **model_kwargs) -> dict:
    """
    Perform K-Fold cross validation and return fold statistics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_mses = []
    val_r2s = []
    train_mses = []
    train_r2s = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        m = model_cls(**model_kwargs)
        m.fit(X_tr, y_tr)
        
        pred_tr = m.predict(X_tr)
        pred_val = m.predict(X_val)
        
        train_mses.append(mean_squared_error(y_tr, pred_tr))
        train_r2s.append(r2_score(y_tr, pred_tr))
        val_mses.append(mean_squared_error(y_val, pred_val))
        val_r2s.append(r2_score(y_val, pred_val))
        
    return {
        'train_mse_mean': float(np.mean(train_mses)),
        'train_mse_std': float(np.std(train_mses)),
        'val_mse_mean': float(np.mean(val_mses)),
        'val_mse_std': float(np.std(val_mses)),
        'val_r2_mean': float(np.mean(val_r2s)),
        'val_r2_std': float(np.std(val_r2s)),
    }


def compute_information_criteria(rss: float, n_samples: int, n_parameters: int) -> dict:
    """
    Compute Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).
    """
    mse = rss / n_samples
    aic = n_samples * np.log(mse) + 2 * n_parameters
    bic = n_samples * np.log(mse) + n_parameters * np.log(n_samples)
    return {'aic': aic, 'bic': bic}
