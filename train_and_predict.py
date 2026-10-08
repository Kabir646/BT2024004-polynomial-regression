#!/usr/bin/env python3
"""
Main training and inference pipeline for Polynomial Regression Assignment.
Roll Number: BT2024004

Generates deliverables:
- BT2024004_pred_var1.csv (Phase 1: Steam Turbine Optimization)
- BT2024004_pred_var2.csv (Phase 2: Thermal Reservoir Mapping)
"""

import os
import argparse
import pandas as pd
import numpy as np
from src.data_loader import load_dataset
from src.models import PolynomialRegressionModel
from src.evaluate import evaluate_predictions, cross_validate_model


def main():
    parser = argparse.ArgumentParser(description="Train polynomial regression models and generate inference.")
    parser.add_argument("--roll_no", type=str, default="BT2024004", help="Student roll number")
    parser.add_argument("--data_dir", type=str, default="BT2024004", help="Directory containing datasets")
    parser.add_argument("--deg_v1", type=int, default=4, help="Polynomial degree for Phase 1 (var1)")
    parser.add_argument("--deg_v2", type=int, default=8, help="Polynomial degree for Phase 2 (var2)")
    parser.add_argument("--model_type", type=str, default="ols", choices=["ols", "ridge", "lasso"], help="Model type to use")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save prediction CSV files")
    args = parser.parse_args()

    print("=" * 70)
    print("POLYNOMIAL REGRESSION ASSIGNMENT - TRAINING & INFERENCE PIPELINE")
    print(f"Student Roll Number : {args.roll_no}")
    print(f"Model Type          : {args.model_type.upper()}")
    print(f"Phase 1 Degree (v1) : {args.deg_v1}")
    print(f"Phase 2 Degree (v2) : {args.deg_v2}")
    print("=" * 70)

    # -------------------------------------------------------------
    # PHASE 1: Steam Turbine Optimization (var1)
    # -------------------------------------------------------------
    print("\n>>> Executing Phase 1: Steam Turbine Optimization (var1)...")
    X1_tr, y1_tr, X1_te, fnames1, df1_tr, df1_te = load_dataset(args.data_dir, args.roll_no, "var1")
    print(f"Loaded Phase 1: Train shape {X1_tr.shape}, Test shape {X1_te.shape}")

    # Cross-validation validation check
    cv_res1 = cross_validate_model(
        PolynomialRegressionModel, X1_tr, y1_tr,
        n_splits=10, random_state=42,
        degree=args.deg_v1, model_type=args.model_type, alpha=10.0 if args.model_type == 'ridge' else 0.01
    )
    print(f"Phase 1 10-Fold CV Validation: MSE = {cv_res1['val_mse_mean']:.5f} +/- {cv_res1['val_mse_std']:.5f}, R2 = {cv_res1['val_r2_mean']:.5f}")

    # Full training
    model1 = PolynomialRegressionModel(degree=args.deg_v1, model_type=args.model_type, alpha=10.0 if args.model_type == 'ridge' else 0.01)
    model1.fit(X1_tr, y1_tr)
    n_params1 = model1.get_num_features()
    cond1 = model1.get_condition_number(X1_tr)
    train_eval1 = evaluate_predictions(y1_tr, model1.predict(X1_tr))
    print(f"Phase 1 Full Train fit: MSE = {train_eval1['mse']:.5f}, R2 = {train_eval1['r2']:.5f}")
    print(f"Phase 1 Parameters    : {n_params1} polynomial features | Condition Number: {cond1:.2e}")

    # Test inference
    pred_v1 = model1.predict(X1_te)
    pred_v1_df = pd.DataFrame({'y': pred_v1})
    out_v1_path = os.path.join(args.output_dir, f"{args.roll_no}_pred_var1.csv")
    pred_v1_df.to_csv(out_v1_path, index=False)
    print(f"Saved Phase 1 predictions to -> {out_v1_path} (Rows: {len(pred_v1_df)})")

    # -------------------------------------------------------------
    # PHASE 2: Subterranean Thermal Reservoir Mapping (var2)
    # -------------------------------------------------------------
    print("\n>>> Executing Phase 2: Subterranean Thermal Reservoir Mapping (var2)...")
    X2_tr, y2_tr, X2_te, fnames2, df2_tr, df2_te = load_dataset(args.data_dir, args.roll_no, "var2")
    print(f"Loaded Phase 2: Train shape {X2_tr.shape}, Test shape {X2_te.shape}")

    # Cross-validation validation check
    cv_res2 = cross_validate_model(
        PolynomialRegressionModel, X2_tr, y2_tr,
        n_splits=10, random_state=42,
        degree=args.deg_v2, model_type=args.model_type, alpha=0.2 if args.model_type == 'ridge' else 0.005
    )
    print(f"Phase 2 10-Fold CV Validation: MSE = {cv_res2['val_mse_mean']:.5f} +/- {cv_res2['val_mse_std']:.5f}, R2 = {cv_res2['val_r2_mean']:.5f}")

    # Full training
    model2 = PolynomialRegressionModel(degree=args.deg_v2, model_type=args.model_type, alpha=0.2 if args.model_type == 'ridge' else 0.005)
    model2.fit(X2_tr, y2_tr)
    n_params2 = model2.get_num_features()
    cond2 = model2.get_condition_number(X2_tr)
    train_eval2 = evaluate_predictions(y2_tr, model2.predict(X2_tr))
    print(f"Phase 2 Full Train fit: MSE = {train_eval2['mse']:.5f}, R2 = {train_eval2['r2']:.5f}")
    print(f"Phase 2 Parameters    : {n_params2} polynomial features | Condition Number: {cond2:.2e}")

    # Test inference
    pred_v2 = model2.predict(X2_te)
    pred_v2_df = pd.DataFrame({'y': pred_v2})
    out_v2_path = os.path.join(args.output_dir, f"{args.roll_no}_pred_var2.csv")
    pred_v2_df.to_csv(out_v2_path, index=False)
    print(f"Saved Phase 2 predictions to -> {out_v2_path} (Rows: {len(pred_v2_df)})")

    # -------------------------------------------------------------
    # SANITY CHECKS & VERIFICATION
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUBMISSION VERIFICATION & QUALITY CHECKS")
    print("=" * 70)
    for path, name in [(out_v1_path, "Phase 1"), (out_v2_path, "Phase 2")]:
        chk_df = pd.read_csv(path)
        assert list(chk_df.columns) == ['y'], f"Column header mismatch in {path}!"
        assert len(chk_df) == 1000, f"Expected 1000 rows in {path}, got {len(chk_df)}!"
        assert not chk_df['y'].isnull().any(), f"NaNs found in {path}!"
        assert not np.isinf(chk_df['y']).any(), f"Infs found in {path}!"
        print(f"[{name}] [OK] Format verified: 1000 rows, header 'y', min={chk_df['y'].min():.4f}, max={chk_df['y'].max():.4f}, mean={chk_df['y'].mean():.4f}")

    print("\nTraining and inference pipeline completed successfully!")


if __name__ == "__main__":
    main()
