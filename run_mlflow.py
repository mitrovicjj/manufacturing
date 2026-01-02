import mlflow
import mlflow.sklearn
import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from src.ml.train import train_model

# MLflow setup
mlflow.set_experiment("predictive_maintenance_production")
print("🚀 MLflow PRODUCTION experiment - 12 runs + business metrics")

# 12 eksperimenata - širok grid search
experiments = [
    # Baseline
    {"name": "base_rw20_tw5_lr008", "rolling_window": 20, "target_window": 5, "lr": 0.08, "depth": 6},
    {"name": "base_rw30_tw8_lr012", "rolling_window": 30, "target_window": 8, "lr": 0.12, "depth": 6},
    
    # Rolling window varijacije
    {"name": "rw15_tw5_lr008", "rolling_window": 15, "target_window": 5, "lr": 0.08, "depth": 6},
    {"name": "rw40_tw10_lr010", "rolling_window": 40, "target_window": 10, "lr": 0.10, "depth": 6},
    
    # Learning rate varijacije
    {"name": "rw25_tw7_lr005", "rolling_window": 25, "target_window": 7, "lr": 0.05, "depth": 6},
    {"name": "rw25_tw7_lr015", "rolling_window": 25, "target_window": 7, "lr": 0.15, "depth": 6},
    
    # Depth varijacije
    {"name": "rw20_tw8_d4", "rolling_window": 20, "target_window": 8, "lr": 0.08, "depth": 4},
    {"name": "rw20_tw8_d8", "rolling_window": 20, "target_window": 8, "lr": 0.08, "depth": 8},
    
    # Target window varijacije (short/medium/long term)
    {"name": "short_rw20_tw3", "rolling_window": 20, "target_window": 3, "lr": 0.08, "depth": 6},
    {"name": "long_rw20_tw12", "rolling_window": 20, "target_window": 12, "lr": 0.08, "depth": 6},
    
    # Aggressive (high recall)
    {"name": "agg_rw30_tw5_lr010_d5", "rolling_window": 30, "target_window": 5, "lr": 0.10, "depth": 5},
    
    # Conservative (high precision)
    {"name": "cons_rw25_tw6_lr006_d7", "rolling_window": 25, "target_window": 6, "lr": 0.06, "depth": 7},
]

data_path = "data/raw/logs_all_machines_v2.csv"

for i, exp in enumerate(experiments):
    print(f"\n{'='*80}")
    print(f"[{i+1}/12] 🧪 Eksperiment: {exp['name']}")
    print('='*80)
    
    with mlflow.start_run(run_name=exp['name']):
        # LOGUJ SVE PARAMETRE
        mlflow.log_param("rolling_window", exp['rolling_window'])
        mlflow.log_param("target_window", exp['target_window'])
        mlflow.log_param("learning_rate", exp['lr'])
        mlflow.log_param("max_depth", exp['depth'])
        mlflow.log_param("data_file", os.path.basename(data_path))
        
        # Model parametri
        model_params = {
            'n_estimators': 200,
            'max_depth': exp['depth'],
            'learning_rate': exp['lr'],
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'scale_pos_weight': 20,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # TRENAJ FULL pipeline
        print("🎯 Training model...")
        clf, X_test, y_test = train_model(
            data_path=data_path,
            output_model_path=f"models/mlflow_{exp['name']}.pkl",
            target_window=exp['target_window'],
            rolling_window=exp['rolling_window'],
            model_params=model_params
        )
        
        # PRODUCTION METRIKE
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Standardne metrike
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # BUSINESS THRESHOLD (0.35 - tvoj izbor)
        y_pred_035 = (y_proba >= 0.35).astype(int)
        recall_035 = recall_score(y_test, y_pred_035)
        precision_035 = precision_score(y_test, y_pred_035, zero_division=0)
        f1_035 = f1_score(y_test, y_pred_035, zero_division=0)
        
        # Specificity (true negative rate)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        # BUSINESS METRIKE
        downtime_cases = (y_test == 1).sum()
        false_alarms = (y_pred == 1).sum() - tp
        downtime_caught = tp / downtime_cases if downtime_cases > 0 else 0
        
        # LOGUJ SVE U MLflow
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("specificity", specificity)
        
        # Threshold 0.35 (BUSINESS)
        mlflow.log_metric("recall_035", recall_035)
        mlflow.log_metric("precision_035", precision_035)
        mlflow.log_metric("f1_035", f1_035)
        
        # Business metrics
        mlflow.log_metric("downtime_cases", downtime_cases)
        mlflow.log_metric("downtime_caught", downtime_caught)
        mlflow.log_metric("false_alarms", false_alarms)
        mlflow.log_param("n_features", len(X_test.columns))
        
        # LOGUJ MODEL
        mlflow.sklearn.log_model(clf, "model", name="xgb_production_pipeline")
        
        print(f"📊 RESULTS: ROC-AUC={roc_auc:.4f} | Recall={recall:.3f} | F1={f1:.3f}")
        print(f"🎯 BUSINESS @0.35: Recall={recall_035:.3f} | Precision={precision_035:.3f}")
        print(f"💾 Model: models/mlflow_{exp['name']}.pkl")
