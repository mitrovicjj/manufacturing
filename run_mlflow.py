#!/usr/bin/env python3
"""
Unified MLflow Experiment Runner - XGBoost + ANFIS + Comparisons
Updated za master prezentaciju: CV, edge cases, model comparison
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.ml.train import create_model_pipeline, handle_imbalance, train_model
from src.anfis.train import (
    ANFISAdvanced,
    convert_to_pytorch,
    train_hybrid,
    evaluate,
    forward_torch,
)

from src.anfis.config import ANFISConfig


# ---------------------------------------------------------------------
# ANFIS PyFunc wrapper
# ---------------------------------------------------------------------
class ANFISPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, anfis_model):
        self.anfis_model = anfis_model

    def predict(self, context, model_input):
        X_torch = torch.tensor(
            model_input.values if hasattr(model_input, "values") else model_input,
            dtype=torch.float32,
        )
        with torch.no_grad():
            y_pred_t = forward_torch(self.anfis_model, X_torch)
        return y_pred_t.cpu().numpy().reshape(-1, 1)


# ---------------------------------------------------------------------
# Shared data preparation (FIXED INDENTATION)
# ---------------------------------------------------------------------
def prepare_data(data_path, rolling_window=30, target_window=8, for_anfis=False):
    """Shared data prep sa feature store"""
    from src.ml.feature_store import FeatureStore
    from src.ml.features import build_features, get_feature_columns
    from sklearn.model_selection import train_test_split

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

    print(f"📊 Features shape: {df_features.shape}")
    print(f"📋 First 10 columns: {df_features.columns.tolist()[:10]}")

    downtime_cols = df_features.filter(like="downtime_next").columns.tolist()
    print(f"🔍 Downtime columns found: {downtime_cols}")

    if not downtime_cols:
        raise ValueError("❌ No 'downtime_next' column found!")

    target_col = downtime_cols[0]
    print(f"✅ Selected target: '{target_col}'")

    feature_cols = get_feature_columns(df_features, target_col, numeric_only=for_anfis)
    print(f"✅ Features selected: {len(feature_cols)} columns")

    numeric_df = df_features[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    X = numeric_df.values.astype(np.float32)
    y = df_features[target_col].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# Main entry point (FIXED STRUCTURE + INDENTATION)
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Master MLflow Runner")
    parser.add_argument(
        "--mode",
        choices=["xgboost_grid", "anfis_single", "edge_nan", "comparison"],
        default="xgboost_grid",
    )
    parser.add_argument("--data_path", default="data/raw/logs_all_machines_v2.csv")
    parser.add_argument("--rolling_window", type=int, default=30)
    parser.add_argument("--target_window", type=int, default=8)
    parser.add_argument("--inject_nan_pct", type=float, default=0.0)
    parser.add_argument(
        "--premise_training",
        type=bool,
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument("--experiment_name", default="default_exp")
    parser.add_argument("--model_type", choices=["xgboost", "anfis", "comparison"], default="xgboost")
    parser.add_argument("--params", type=str, default='{}')  # JSON string

    args = parser.parse_args()

    print(f"🚀 Mode: {args.mode} | Data: {args.data_path}")

    # -----------------------------------------------------------------
    # XGBOOST GRID - COMPLETE WORKING VERSION
    # -----------------------------------------------------------------
    if args.mode == "xgboost_grid":
        mlflow.set_experiment("predictive_maintenance_production")
        
        experiments = [
            {"name": "base_rw20_tw5_lr008", "rolling_window": 20, "target_window": 5, "lr": 0.08, "depth": 6},
            {"name": "deep_rw30_tw8_lr005", "rolling_window": 30, "target_window": 8, "lr": 0.05, "depth": 8},
        ]
        
        for exp in experiments:
            print(f"\n🚀 Running experiment: {exp['name']}")
            
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
                
                # ✅ SVE radi - pipeline ce se automatski fit-ovati
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
                print(f"✅ {exp['name']} → MLflow logged!")



    # -----------------------------------------------------------------
    # ANFIS SINGLE
    # -----------------------------------------------------------------
    elif args.mode == "anfis_single":
        mlflow.set_experiment("anfis_hybrid_study")

        X_train, X_test, y_train, y_test = prepare_data(
            args.data_path,
            args.rolling_window,
            args.target_window,
        )

        if args.inject_nan_pct > 0:
            nan_mask = np.random.rand(*X_train.shape) < args.inject_nan_pct
            X_train[nan_mask] = np.nan
            mlflow.log_param("inject_nan_pct", args.inject_nan_pct)

        with mlflow.start_run(
            run_name=f"anfis_rw{args.rolling_window}_tw{args.target_window}"
        ):
            params = {
                "epochs": 100,
                "lr_premise": 1e-3,
                "lr_consequent": 1e-2,
            }
            mlflow.log_params(params)

            model = ANFISAdvanced(config=ANFISConfig())
            convert_to_pytorch(model)

            history = train_hybrid(model, X_train, y_train, **params)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_metric("final_loss", history["loss"][-1])
            mlflow.log_metrics(metrics)

            pyfunc = ANFISPyFunc(model)
            mlflow.pyfunc.log_model(
                "anfis_model",
                python_model=pyfunc,
                input_example=X_test[:5],
            )

    # -----------------------------------------------------------------
    # EDGE CASE MODE (NO RECURSION)
    # -----------------------------------------------------------------
    elif args.mode == "edge_nan":
        mlflow.set_experiment("edge_cases")
        args.inject_nan_pct = 0.20
        print(
            "Edge case mode enabled. "
            "Run xgboost_grid with --inject_nan_pct 0.20"
        )

    print("✅ All experiments logged! Check MLflow UI.")


if __name__ == "__main__":
    main()