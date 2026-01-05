import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from src.ml.features import build_features, get_feature_columns
from src.ml.feature_store import FeatureStore
from hashlib import md5
import mlflow.xgboost

def create_model_pipeline(df_features, target_col, model_params=None):
    """
    FULL sklearn Pipeline: numeric + categorical encoded features + XGBoost.
    
    Args:
        df_features: DataFrame sa svim feature kolonama
        target_col: Naziv target kolone (npr. 'downtime_next_5')
        model_params: Dict of XGBoost hyperparameters (optional)
    
    Returns:
        sklearn Pipeline spreman za fit()
    """
    
    if model_params is None:
        model_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'scale_pos_weight': 20,
            'eval_metric': 'logloss',
            'random_state': 42
        }
    
    # 1. Automatski feature selection
    feature_cols = get_feature_columns(df_features, target_col)
    X_sample = df_features[feature_cols]
    
    # 2. NUMERIC features
    numeric_features = X_sample.select_dtypes(include=['number']).columns.tolist()
    
    # 3. CATEGORICAL encoded features (za OneHotEncoder)
    cat_encoded_features = [col for col in feature_cols 
                          if any(x in col.lower() for x in ['_encoded', '_state_encoded'])]
    
    print(f"Pipeline: {len(numeric_features)} numeric + {len(cat_encoded_features)} cat → {len(feature_cols)} total")
    
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    
    # 4. FULL preprocessing
    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), cat_encoded_features)
    ], remainder='drop')  # Drop sve ostalo
    
    # 5. Pipeline
    model = XGBClassifier(**model_params)
    clf = Pipeline([
        ('preprocess', preprocess),
        ('model', model)
    ])
    
    return clf

def handle_imbalance(X_train, y_train, method='oversample', sampling_strategy='auto'):
    """
    Handle class imbalance with oversampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        method: 'oversample' or 'none'
        sampling_strategy: 'auto' or ratio (e.g., 0.5)
    
    Returns:
        X_resampled, y_resampled
    """
    if method == 'oversample':
        print(f"Class distribution before oversampling:")
        print(f"  Negative: {(y_train == 0).sum()}")
        print(f"  Positive: {(y_train == 1).sum()}")
        
        oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        
        print(f"Class distribution after oversampling:")
        print(f"  Negative: {(y_resampled == 0).sum()}")
        print(f"  Positive: {(y_resampled == 1).sum()}")
        
        return X_resampled, y_resampled
    
    elif method == 'none':
        print("No oversampling applied.")
        return X_train, y_train
    
    else:
        raise ValueError(f"Unknown method: {method}")

# TRAINING FUNCTION

def train_model(data_path, 
                output_model_path,
                output_data_path=None,
                target_type='windowed',
                target_window=5,
                rolling_window=20,
                test_size=0.2,
                oversample=True,
                model_params=None,
                inject_nan_pct=0.0):
    """
    Complete training pipeline sa FULL features (numeric + categorical encoded).
    
    Args:
        data_path: Path to raw CSV data
        output_model_path: Path to save trained model (.pkl)
        output_data_path: Optional path to save feature-engineered data
        target_type: 'next' or 'windowed'
        target_window: If windowed, window size
        rolling_window: Rolling window for features
        test_size: Train-test split ratio
        oversample: Whether to apply oversampling
        model_params: Dict of XGBoost params (optional)
    
    Returns:
        Trained pipeline, X_test, y_test
    """
    print("="*70)
    print("TRAINING PIPELINE START - FULL FEATURES")
    print("="*70)
    
    # [1/6] Load data
    print(f"\n[1/6] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # [2/6] Feature engineering SA CACHE-OM
    print(f"\n[2/6] Feature engineering (sa cache-om)...")
    store = FeatureStore()

    feature_config = {
        'data_path': os.path.basename(data_path),
        'target_type': target_type,
        'target_window': target_window,
        'rolling_window': rolling_window
    }
    df_features = store.get_features(feature_config)

    if df_features is None:
        print("  Računam features...")
        df_features = build_features(
            df, target_type=target_type, 
            target_window=target_window, 
            rolling_window=rolling_window
        )
        store.save_features(df_features, feature_config)
    else:
        print("  ✓ Features učitani iz cache-a!")
    
    # Save feature-engineered data if requested
    if output_data_path:
        df_features.to_csv(output_data_path, index=False)
        print(f"  ✓ Saved feature data to: {output_data_path}")
    
    # [3/6] Prepare features + target
    print(f"\n[3/6] Preparing features and target...")
    target_col = 'downtime_next' if target_type == 'next' else f'downtime_next_{target_window}'
    
    feature_cols = get_feature_columns(df_features, target_col=target_col)
    X = df_features[feature_cols]
    y = df_features[target_col]
    
    print(f"  Features: {len(feature_cols)} columns (SVE features!)")
    print(f"  Target: {target_col}")
    print(f"  Class balance: {(y==0).sum()} negative, {(y==1).sum()} positive ({(y==1).mean()*100:.1f}% positive)")
    
    # [4/6] Train-test split (CELO df_features za pipeline)
    print(f"\n[4/6] Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_features, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # [5/6] Handle imbalance (oversampling SAMO train set)
    print(f"\n[5/6] Handling class imbalance...")
    if oversample:
        X_train_resampled, y_train_resampled = handle_imbalance(
            X_train[feature_cols], y_train, method='oversample'
        )
    else:
        X_train_resampled, y_train_resampled = handle_imbalance(
            X_train[feature_cols], y_train, method='none'
        )
    
    if inject_nan_pct > 0:
        print(f"💉 Injecting {inject_nan_pct*100}% NaNs into training data...")
        nan_mask = np.random.rand(*X_train_resampled.shape) < inject_nan_pct
        X_train_resampled[nan_mask] = np.nan
        mlflow.log_param("inject_nan_pct", inject_nan_pct)


    # [6/6] FULL Pipeline training
    print(f"\n[6/6] Training FULL pipeline...")
    clf = create_model_pipeline(df_features, target_col, model_params=model_params)
    clf.fit(X_train_resampled, y_train_resampled)  # Pipeline prima df_features!
    print("  ✓ Model training complete!")
    
    # Save model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(clf, output_model_path)
    print(f"  ✓ Model saved to: {output_model_path}")
    
    # Quick evaluation
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    y_pred = clf.predict(X_test[feature_cols])
    y_proba = clf.predict_proba(X_test[feature_cols])[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("\n" + "="*70)
    print("QUICK EVALUATION ON TEST SET")
    print("="*70)
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("="*70)
    
    return clf, X_test[feature_cols], y_test


# CLI

def main():
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    
    # Required arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to raw CSV data")
    parser.add_argument("--model_output", type=str, required=True,
                        help="Path to save trained model (.pkl)")
    
    # Optional arguments
    parser.add_argument("--data_output", type=str, default=None,
                        help="Path to save feature-engineered data (optional)")
    parser.add_argument("--target_type", type=str, default='windowed',
                        choices=['next', 'windowed'],
                        help="Target type: 'next' or 'windowed'")
    parser.add_argument("--target_window", type=int, default=5,
                        help="Window size for windowed target")
    parser.add_argument("--rolling_window", type=int, default=20,
                        help="Rolling window size for features")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test set proportion")
    parser.add_argument("--no_oversample", action='store_true',
                        help="Disable oversampling")
    
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    
    args = parser.parse_args()
    
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'scale_pos_weight': 20,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    with mlflow.start_run(run_name=f"xgb_{args.target_window}w_{args.rolling_window}rw"):  # ← DODAJ
        mlflow.log_params(vars(args))  # Loguje sve CLI args
        mlflow.log_param("model_params", model_params)

    train_model(
        data_path=args.data,
        output_model_path=args.model_output,
        output_data_path=args.data_output,
        target_type=args.target_type,
        target_window=args.target_window,
        rolling_window=args.rolling_window,
        test_size=args.test_size,
        oversample=not args.no_oversample,
        model_params=model_params
    )


if __name__ == "__main__":
    main()