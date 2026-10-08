"""
Polynomial Regression Model implementations (OLS, Ridge, Lasso).
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline


class PolynomialRegressionModel:
    """
    Wrapper for Polynomial Regression with optional L1/L2 regularization and feature scaling.
    """
    def __init__(self, degree: int, model_type: str = 'ols', alpha: float = 1.0, include_bias: bool = True):
        """
        Args:
            degree: Maximum polynomial degree.
            model_type: 'ols', 'ridge', or 'lasso'.
            alpha: Regularization strength for Ridge/Lasso.
            include_bias: Whether to include bias term.
        """
        self.degree = degree
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.include_bias = include_bias
        
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        self.scaler = None
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit polynomial model on X and y.
        """
        X_poly = self.poly.fit_transform(X)
        
        if self.model_type == 'ols':
            # For OLS, fit directly without intercept if bias already included
            self.model = LinearRegression(fit_intercept=not self.include_bias)
            self.model.fit(X_poly, y)
        elif self.model_type == 'ridge':
            self.scaler = StandardScaler()
            # If bias included, scale columns excluding bias
            if self.include_bias:
                X_scaled = self.scaler.fit_transform(X_poly[:, 1:])
                self.model = Ridge(alpha=self.alpha, fit_intercept=True)
                self.model.fit(X_scaled, y)
            else:
                X_scaled = self.scaler.fit_transform(X_poly)
                self.model = Ridge(alpha=self.alpha, fit_intercept=True)
                self.model.fit(X_scaled, y)
        elif self.model_type == 'lasso':
            self.scaler = StandardScaler()
            if self.include_bias:
                X_scaled = self.scaler.fit_transform(X_poly[:, 1:])
                self.model = Lasso(alpha=self.alpha, max_iter=20000, fit_intercept=True)
                self.model.fit(X_scaled, y)
            else:
                X_scaled = self.scaler.fit_transform(X_poly)
                self.model = Lasso(alpha=self.alpha, max_iter=20000, fit_intercept=True)
                self.model.fit(X_scaled, y)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Must be 'ols', 'ridge', or 'lasso'.")
            
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for X.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
            
        X_poly = self.poly.transform(X)
        
        if self.model_type == 'ols':
            return self.model.predict(X_poly)
        else:
            if self.include_bias:
                X_scaled = self.scaler.transform(X_poly[:, 1:])
                return self.model.predict(X_scaled)
            else:
                X_scaled = self.scaler.transform(X_poly)
                return self.model.predict(X_scaled)

    def get_num_features(self) -> int:
        """
        Get total number of polynomial features.
        """
        return self.poly.n_output_features_

    def get_condition_number(self, X: np.ndarray) -> float:
        """
        Compute condition number of the polynomial design matrix.
        """
        X_poly = self.poly.transform(X)
        return float(np.linalg.cond(X_poly))
