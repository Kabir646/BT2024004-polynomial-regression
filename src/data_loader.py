"""
Data loading and preprocessing utilities for Polynomial Regression Assignment.
Student Roll Number: BT2024004
"""

import os
import pandas as pd
import numpy as np


def load_dataset(data_dir: str, roll_no: str, var_id: str):
    """
    Load train and test datasets for a given roll number and problem ID.
    
    Args:
        data_dir: Directory containing the CSV datasets.
        roll_no: Student roll number (e.g., 'BT2024004').
        var_id: Problem variant ('var1' or 'var2').
        
    Returns:
        tuple: (X_train, y_train, X_test, feature_names, df_train, df_test)
    """
    train_filename = f"{roll_no}_train_{var_id}.csv"
    test_filename = f"{roll_no}_test_{var_id}.csv"
    
    train_path = os.path.join(data_dir, train_filename)
    test_path = os.path.join(data_dir, test_filename)
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    if var_id == "var1":
        feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    elif var_id == "var2":
        feature_names = ['x1', 'x2', 'x3']
    else:
        raise ValueError(f"Unknown var_id: {var_id}. Expected 'var1' or 'var2'.")
        
    X_train = df_train[feature_names].values
    y_train = df_train['y'].values
    X_test = df_test[feature_names].values
    
    return X_train, y_train, X_test, feature_names, df_train, df_test
