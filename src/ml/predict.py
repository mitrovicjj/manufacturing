import os
import argparse
import joblib
import pandas as pd
import numpy as np
from config import (
    FINAL_MODEL_PATH,
    FINAL_THRESHOLD,
    FINAL_PREDICTIONS_PATH)
from src.ml.features import build_features, get_feature_columns

def load_model(model_path):
    """
    Load trained model from disk.
        model_path: path to .pkl model file
    Returns:
        Loaded model pipeline
    """
    model = joblib.load(model_path)
    # set n_jobs to 1 for reproducibility
    if hasattr(model, 'n_jobs'):
        model.n_jobs = 1
    return model

def predict(model, X, threshold=0.5):
    """
    Generate predictions and probabilities.
    Returns:
        probs: Predicted probabilities for positive class
        preds: Binary predictions (0 or 1)
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    preds = (probs >= threshold).astype(int)
    return probs, preds

def predict_on_data(model_path,
                    data_path,
                    output_path=None,
                    target_type='windowed',
                    target_window=5,
                    rolling_window=20,
                    target_col=None):
    """
    Complete prediction pipeline on new data.
    
    Args:
        model_path: path to .pkl model file
        data_path: path to raw CSV data
        output_path: path to save predictions-optional
        target_type: 'next' or 'windowed' (must match training)
        target_window: window size-optional
        rolling_window: rolling window size (must match training)
        target_col: target column name if data has labels (optional)
    
    Returns:
        DF with predictions and probabilities
    """

    print("="*70)
    print("PREDICTION PIPELINE START")
    print("="*70)
    
    print(f"\n[1/4] Loading model from: {model_path}")
    model = load_model(model_path)
    print("     Model loaded")
    
    print(f"\n[2/4] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    print(f"\n[3/4] Feature engineering...")
    print(f"  (Must match training config: target_type={target_type}, rolling_window={rolling_window})")
    
    df_features = build_features(
        df,
        target_type=target_type,
        target_window=target_window,
        rolling_window=rolling_window
    )
    
    # determine target column name
    if target_col is None:
        target_col = 'downtime_next' if target_type == 'next' else f'downtime_next_{target_window}'
    
    # get feature columns (same as training)
    feature_cols = get_feature_columns(df_features, target_col=target_col)
    X = df_features[feature_cols]
    
    print(f"  Features: {len(feature_cols)} columns")
    
    print(f"\n[4/4] Generating predictions...")
    probs, preds = predict(model, X)
    
    # add predictions to dataframe
    df_features['downtime_risk'] = probs
    df_features['downtime_predicted'] = preds
    
    print(f"  ✓ Predictions generated for {len(df_features)} rows")
    print(f"  Predicted downtime rate: {preds.mean()*100:.2f}%")
    print(f"  Average risk score: {probs.mean():.4f}")
    
    if output_path:
        df_features.to_csv(output_path, index=False)
        print(f"  ✓ Predictions saved to: {output_path}")
    
    print("="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    
    return df_features


# BATCH PREDICTION FOR SIMULATION

def predict_final_for_simulation(
    data_path,
    output_path=FINAL_PREDICTIONS_PATH,
    target_type='windowed',
    target_window=5,
    rolling_window=30,
    lag_periods=[1,3,5]
):
    """
    Final frozen inference using FINAL_MODEL_PATH and FINAL_THRESHOLD.
    Performs feature engineering in-place.
    """
    print("\nRUNNING FINAL MODEL FOR SIMULATION")

    model = load_model(FINAL_MODEL_PATH)
    df = pd.read_csv(data_path)

    df_features = build_features(
        df,
        target_type=target_type,
        target_window=target_window,
        rolling_window=rolling_window
    )


    target_col = f'downtime_next_{target_window}'
    feature_cols = get_feature_columns(df_features, target_col=target_col)
    X = df_features[feature_cols]

    probs, preds = predict(model, X, threshold=FINAL_THRESHOLD)

    df_features['downtime_risk'] = probs
    df_features['downtime_predicted'] = preds

    if 'cycle_id' not in df_features.columns:
        df_features['cycle_id'] = range(len(df_features))

    df_out = df_features[['cycle_id', 'downtime_risk', 'downtime_predicted']]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)

    metrics = {
        'threshold': FINAL_THRESHOLD,
        'predicted_downtime_rate': preds.mean(),
        'average_risk_score': probs.mean(),
        'num_rows': len(df_features),
        'num_features': len(df_features.columns)
    }
    metrics_path = os.path.join(os.path.dirname(output_path), "metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    #fll_features_path = os.path.join(os.path.dirname(output_path), "features_with_predictions.csv")
    #f_features.to_csv(full_features_path, index=False)

    print(f"✓ Final predictions saved to {output_path}")
    print(f"✓ Threshold used: {FINAL_THRESHOLD}")
    print(f"✓ Predicted downtime rate: {preds.mean()*100:.2f}%")

    return df_out


# CLI

def main():
    parser = argparse.ArgumentParser(description="Predict downtime with trained model")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pkl)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to raw CSV data")
    
    # Optional arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save predictions (optional)")
    parser.add_argument("--target_type", type=str, default='windowed',
                        choices=['next', 'windowed'],
                        help="Target type (must match training)")
    parser.add_argument("--target_window", type=int, default=5,
                        help="Target window (must match training)")
    parser.add_argument("--rolling_window", type=int, default=20,
                        help="Rolling window (must match training)")
    parser.add_argument("--for_simulation", action='store_true',
                        help="Output format for Tecnomatix simulation")
    
    args = parser.parse_args()
    
    if args.for_simulation:
        predict_on_data(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            target_type=args.target_type,
            target_window=args.target_window,
            rolling_window=args.rolling_window
        )


if __name__ == "__main__":
    main()