#!/usr/bin/env python3
"""
Unified MLflow Experiment Runner - XGBoost + ANFIS + Comparisons
Optimized: Faster training, better metrics, feature name tracking
"""

import os
import argparse
from typing import Any, Union

import numpy as np
import pandas as pd
import torch

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.ml.train import train_model
from src.ml.feature_store import FeatureStore
from src.ml.features import build_features, get_feature_columns

from src.anfis.config import ANFISConfig
from src.anfis.core import ANFISAdvanced
from src.anfis.train import (
    convert_to_pytorch,
    train_hybrid,
    evaluate,
    forward_torch,
)


# =============================================================================
# MLflow PyFunc Wrapper
# =============================================================================

class ANFISPyFunc(mlflow.pyfunc.PythonModel):
    """ANFIS model wrapper for MLflow deployment."""
    
    def __init__(self, anfis_model):
        self.anfis_model = anfis_model

    def predict(
        self,
        context: Any,
        model_input: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Predict using ANFIS model.
        
        Args:
            context: MLflow model context (unused)
            model_input: Input features as DataFrame or numpy array
                        Shape: (n_samples, n_features)
            
        Returns:
            Predictions as numpy array with shape (n_samples, 1)
        """
        X_torch = torch.tensor(
            model_input.values if hasattr(model_input, "values") else model_input,
            dtype=torch.float32,
        )
        
        with torch.no_grad():
            y_pred_t = forward_torch(self.anfis_model, X_torch)
        
        return y_pred_t.cpu().numpy().reshape(-1, 1)


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_data(data_path, rolling_window=30, target_window=8):
    """
    Prepare training and test data with feature engineering.
    
    Args:
        data_path: Path to raw data CSV
        rolling_window: Window size for rolling features
        target_window: Window size for target creation
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    store = FeatureStore()
    config = {
        "data_path": os.path.basename(data_path),
        "rolling_window": rolling_window,
        "target_window": target_window,
    }

    df_features = store.get_features(config)
    if df_features is None:
        df_raw = pd.read_csv(data_path)
        df_features = build_features(
            df_raw,
            rolling_window=rolling_window,
            target_window=target_window,
        )
        store.save_features(df_features, config)

    print(f"Features shape: {df_features.shape}")
    print(f"First 10 columns: {df_features.columns.tolist()[:10]}")

    downtime_cols = df_features.filter(like="downtime_next").columns.tolist()
    print(f"Downtime columns found: {downtime_cols}")

    if not downtime_cols:
        raise ValueError("No 'downtime_next' column found!")

    target_col = f"downtime_next_{target_window}"
    if target_col not in df_features.columns:
        target_col = downtime_cols[0]
        print(f"WARNING: {target_col} not found, using {downtime_cols[0]}")

    print(f"Selected target: '{target_col}' (window={target_window})")

    feature_cols = get_feature_columns(df_features, target_col, numeric_only=True)
    print(f"Features selected: {len(feature_cols)} columns")

    numeric_df = df_features[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    feature_names = numeric_df.columns.tolist()
    
    X = numeric_df.values.astype(np.float32)
    y = df_features[target_col].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Loaded X_train shape: {X_train.shape}")
    print(f"Loaded y_train shape: {y_train.shape}")
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, feature_names


# =============================================================================
# Feature Selection
# =============================================================================

def select_top_features_for_anfis(
    X_train, y_train, X_test, feature_names, n_features=8, n_mfs=2
):
    """
    Select top N features using Random Forest importance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        feature_names: List of feature names
        n_features: Number of features to select
        n_mfs: Number of membership functions per feature
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_feature_names)
    """
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION FOR ANFIS")
    print(f"{'='*70}")
    print(f"   Original features: {X_train.shape[1]}")
    print(f"   Target features: {n_features}")
    print(f"   MFs per feature: {n_mfs}")
    
    print("   Computing feature importance...")
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(n_features)['feature'].tolist()
    
    print(f"\nTOP {n_features} FEATURES SELECTED:")
    for i, (feat, imp) in enumerate(
        zip(top_features, importance_df.head(n_features)['importance']), 1
    ):
        print(f"   {i:2d}. {feat:35s} -> {imp:.4f}")
    
    indices = [feature_names.index(f) for f in top_features]
    X_train_sel = X_train[:, indices]
    X_test_sel = X_test[:, indices]
    
    print(f"Selected shape: {X_train_sel.shape}")
    expected_rules = n_mfs ** n_features
    print(f"Expected ANFIS rules: {n_mfs}^{n_features} = {expected_rules:,}")
    print(f"{'='*70}\n")
    
    return X_train_sel, X_test_sel, top_features


# =============================================================================
# ANFIS Evaluation with Classification Metrics
# =============================================================================

def evaluate_anfis_with_classification(model, X_test, y_test):
    """
    Evaluate ANFIS with both regression and classification metrics.
    
    Args:
        model: Trained ANFIS model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = evaluate(model, X_test, y_test)
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred_t = forward_torch(model, X_tensor)
    y_pred = y_pred_t.cpu().numpy()
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_test_binary = (y_test > 0.5).astype(int)
    
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    
    print("\n" + "="*70)
    print("CLASSIFICATION METRICS (threshold=0.5)")
    print("="*70)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print("\nClassification Report:")
    print(classification_report(
        y_test_binary,
        y_pred_binary,
        target_names=['No Downtime', 'Downtime']
    ))
    print("="*70)
    
    if len(np.unique(y_test_binary)) > 1:
        metrics['precision'] = precision_score(
            y_test_binary, y_pred_binary, zero_division=0
        )
        metrics['recall'] = recall_score(
            y_test_binary, y_pred_binary, zero_division=0
        )
        metrics['f1'] = f1_score(
            y_test_binary, y_pred_binary, zero_division=0
        )
    
    return metrics


# =============================================================================
# XGBoost Grid Search
# =============================================================================

def run_xgboost_grid(args):
    """Run XGBoost grid search experiments."""
    mlflow.set_experiment("predictive_maintenance_production")
    
    experiments = [
        {
            "name": "base_rw20_tw5_lr008",
            "rolling_window": 20,
            "target_window": 5,
            "lr": 0.08,
            "depth": 6
        },
        {
            "name": "deep_rw30_tw8_lr005",
            "rolling_window": 30,
            "target_window": 8,
            "lr": 0.05,
            "depth": 8
        },
    ]
    
    for exp in experiments:
        print(f"\nRunning experiment: {exp['name']}")
        
        with mlflow.start_run(run_name=exp["name"]):
            mlflow.log_params(exp)
            
            model_params = {
                "n_estimators": 200,
                "max_depth": exp["depth"],
                "learning_rate": exp["lr"],
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "scale_pos_weight": 20,
                "eval_metric": "logloss",
                "random_state": 42,
            }
            
            clf, X_test_final, y_test = train_model(
                data_path=args.data_path,
                output_model_path=f"models/mlflow_{exp['name']}.pkl",
                target_window=exp["target_window"],
                rolling_window=exp["rolling_window"],
                test_size=0.2,
                oversample=True,
                model_params=model_params,
                inject_nan_pct=args.inject_nan_pct,
            )
            
            mlflow.sklearn.log_model(clf, "xgboost_pipeline")
            print(f"Experiment {exp['name']} logged to MLflow")


# =============================================================================
# ANFIS Single Run
# =============================================================================

def run_anfis_single(args):
    """Run single ANFIS training experiment."""
    mlflow.set_experiment("anfis_hybrid_optimized")

    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        args.data_path,
        args.rolling_window,
        args.target_window,
    )

    N_ANFIS_FEATURES = args.n_anfis_features
    N_MFS = args.n_mfs
    
    X_train, X_test, selected_features = select_top_features_for_anfis(
        X_train, y_train, X_test, feature_names,
        n_features=N_ANFIS_FEATURES,
        n_mfs=N_MFS
    )

    print(f"\nComputing dynamic feature ranges...")
    feature_ranges = []
    for i in range(X_train.shape[1]):
        feat_min = float(X_train[:, i].min())
        feat_max = float(X_train[:, i].max())
        padding = (feat_max - feat_min) * 0.05
        feature_ranges.append((feat_min - padding, feat_max + padding))
        print(f"   {selected_features[i]:35s} [{feat_min:.4f}, {feat_max:.4f}]")

    if args.inject_nan_pct > 0:
        nan_mask = np.random.rand(*X_train.shape) < args.inject_nan_pct
        X_train[nan_mask] = np.nan
        print(f"Injected {args.inject_nan_pct*100:.1f}% NaN values")

    anfis_config = ANFISConfig(
        n_inputs=N_ANFIS_FEATURES,
        n_mfs_per_input=[N_MFS] * N_ANFIS_FEATURES,
        feature_ranges=feature_ranges,
        use_domain_knowledge=False
    )
    
    anfis_config.validate_config()
    anfis_config.print_config()

    with mlflow.start_run(
        run_name=f"anfis_feat{N_ANFIS_FEATURES}_mf{N_MFS}_ep{args.epochs}_bs{args.batch_size}"
    ):
        mlflow_params = {
            "epochs": args.epochs,
            "lr_premise": 1e-3,
            "lr_consequent": 1e-2,
            "n_features": N_ANFIS_FEATURES,
            "n_mfs_per_feature": N_MFS,
            "n_rules": N_MFS**N_ANFIS_FEATURES,
            "premise_training": args.premise_training,
            "rolling_window": args.rolling_window,
            "target_window": args.target_window,
            "batch_size": args.batch_size,
            "selected_features": ", ".join(selected_features),
        }
        mlflow.log_params(mlflow_params)

        train_params = {
            "epochs": args.epochs,
            "lr_premise": 1e-3,
            "lr_consequent": 1e-2,
            "premise_training": args.premise_training,
            "batch_size": args.batch_size,
            "verbose": True,
        }

        model = ANFISAdvanced(config=anfis_config)
        convert_to_pytorch(model)

        print("\n" + "="*70)
        print("PRE-TRAINING VALIDATION")
        print("="*70)
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Model n_inputs: {model.n_inputs}")
        print(f"Model n_rules: {model.n_rules}")
        print(f"Expected rules: {N_MFS}^{N_ANFIS_FEATURES} = {N_MFS**N_ANFIS_FEATURES:,}")
        print(f"Estimated training time: ~{(args.epochs * 11999 / args.batch_size / 60):.1f} minutes")
        print("="*70)

        assert X_train.shape[1] == model.n_inputs

        history = train_hybrid(model, X_train, y_train, **train_params)
        metrics = evaluate_anfis_with_classification(model, X_test, y_test)

        mlflow.log_metric("final_loss", history["loss"][-1])
        mlflow.log_metrics(metrics)

        pyfunc = ANFISPyFunc(model)
        mlflow.pyfunc.log_model(
            "anfis_model",
            python_model=pyfunc,
            input_example=X_test[:5],
        )

        print(f"ANFIS experiment logged to MLflow")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="MLflow Experiment Runner - XGBoost and ANFIS"
    )
    parser.add_argument(
        "--mode",
        choices=["xgboost_grid", "anfis_single", "anfis_grid", "edge_nan"],
        default="xgboost_grid",
        help="Experiment mode to run"
    )
    parser.add_argument(
        "--data_path",
        default="data/raw/logs_all_machines_v2.csv",
        help="Path to input data CSV"
    )
    parser.add_argument(
        "--rolling_window",
        type=int,
        default=30,
        help="Rolling window size for features"
    )
    parser.add_argument(
        "--target_window",
        type=int,
        default=8,
        help="Target window size"
    )
    parser.add_argument(
        "--inject_nan_pct",
        type=float,
        default=0.0,
        help="Percentage of NaN values to inject (0.0-1.0)"
    )
    parser.add_argument(
        "--premise_training",
        action="store_true",
        default=True,
        help="Enable premise parameter training"
    )
    parser.add_argument(
        "--experiment_name",
        default="default_exp",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--n_anfis_features",
        type=int,
        default=8,
        help="Number of features for ANFIS"
    )
    parser.add_argument(
        "--n_mfs",
        type=int,
        default=2,
        help="Number of membership functions per feature"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training"
    )

    args = parser.parse_args()

    print(f"Mode: {args.mode} | Data: {args.data_path}")

    if args.mode == "xgboost_grid":
        run_xgboost_grid(args)
    
    elif args.mode == "anfis_single":
        run_anfis_single(args)
    
    elif args.mode == "anfis_grid":
        mlflow.set_experiment("anfis_grid_search")
        grid = [
            {"n_features": 6, "n_mfs": 2, "epochs": 30, "batch_size": 256},
            {"n_features": 8, "n_mfs": 2, "epochs": 30, "batch_size": 256},
            {"n_features": 8, "n_mfs": 3, "epochs": 50, "batch_size": 128},
        ]
        
        for config in grid:
            print(f"\n{'='*70}")
            print(f"GRID SEARCH: {config}")
            print(f"{'='*70}")
            print(
                f"Run: python run_mlflow.py --mode anfis_single "
                f"--n_anfis_features {config['n_features']} "
                f"--n_mfs {config['n_mfs']} "
                f"--epochs {config['epochs']} "
                f"--batch_size {config['batch_size']}"
            )
    
    elif args.mode == "edge_nan":
        mlflow.set_experiment("edge_cases")
        print("Edge case mode enabled. Run with --inject_nan_pct 0.20 for NaN testing")

    print("All experiments logged. Check MLflow UI.")


if __name__ == "__main__":
    main()
