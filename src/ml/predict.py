import os
import argparse
import joblib
import pandas as pd
import numpy as np

from src.ml.features import build_features, get_feature_columns

# LOAD MODEL

def load_model(model_path):
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to .pkl model file
    
    Returns:
        Loaded model pipeline
    """
    model = joblib.load(model_path)
    # Set n_jobs to 1 for reproducibility
    if hasattr(model, 'n_jobs'):
        model.n_jobs = 1
    return model

# PREDICT

def predict(model, X):
    """
    Generate predictions and probabilities.
    
    Args:
        model: Trained model pipeline
        X: Features DataFrame
    
    Returns:
        probs: Predicted probabilities for positive class
        preds: Binary predictions (0 or 1)
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)
    
    preds = (probs > 0.5).astype(int)
    
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
        model_path: Path to trained model (.pkl)
        data_path: Path to raw CSV data
        output_path: Path to save predictions (optional)
        target_type: 'next' or 'windowed' (must match training!)
        target_window: Window size if windowed
        rolling_window: Rolling window size (must match training!)
        target_col: Target column name if data has labels (optional)
    
    Returns:
        DataFrame with predictions and probabilities added
    """
    print("="*70)
    print("PREDICTION PIPELINE START")
    print("="*70)
    
    # 1. Load model
    print(f"\n[1/4] Loading model from: {model_path}")
    model = load_model(model_path)
    print("  ✓ Model loaded")
    
    # 2. Load data
    print(f"\n[2/4] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # 3. Feature engineering
    print(f"\n[3/4] Feature engineering...")
    print(f"  (Must match training config: target_type={target_type}, rolling_window={rolling_window})")
    
    df_features = build_features(
        df,
        target_type=target_type,
        target_window=target_window,
        rolling_window=rolling_window
    )
    
    # Determine target column name
    if target_col is None:
        target_col = 'downtime_next' if target_type == 'next' else f'downtime_next_{target_window}'
    
    # Get feature columns (same as training)
    feature_cols = get_feature_columns(df_features, target_col=target_col)
    X = df_features[feature_cols]
    
    print(f"  Features: {len(feature_cols)} columns")
    
    # 4. Predict
    print(f"\n[4/4] Generating predictions...")
    probs, preds = predict(model, X)
    
    # Add predictions to dataframe
    df_features['downtime_risk'] = probs
    df_features['downtime_predicted'] = preds
    
    print(f"  ✓ Predictions generated for {len(df_features)} rows")
    print(f"  Predicted downtime rate: {preds.mean()*100:.2f}%")
    print(f"  Average risk score: {probs.mean():.4f}")
    
    # Save if output path provided
    if output_path:
        df_features.to_csv(output_path, index=False)
        print(f"  ✓ Predictions saved to: {output_path}")
    
    print("="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    
    return df_features


# BATCH PREDICTION FOR SIMULATION

def predict_for_simulation(model_path,
                          data_path,
                          output_path,
                          target_type='windowed',
                          target_window=5,
                          rolling_window=20):
    """
    Predict downtime risk for Tecnomatix simulation integration.
    Outputs CSV with cycle_id and downtime_risk.
    
    Args:
        model_path: Path to trained model
        data_path: Path to raw data
        output_path: Path to save simulation-ready CSV
        target_type: Target type (must match training)
        target_window: Target window (must match training)
        rolling_window: Rolling window (must match training)
    
    Returns:
        DataFrame with predictions
    """
    print("\n📊 GENERATING PREDICTIONS FOR TECNOMATIX SIMULATION")
    
    df_pred = predict_on_data(
        model_path=model_path,
        data_path=data_path,
        output_path=None,  # Don't save full data yet
        target_type=target_type,
        target_window=target_window,
        rolling_window=rolling_window
    )
    
    # Create simulation-ready output
    sim_cols = ['cycle_id', 'downtime_risk', 'downtime_predicted']
    
    # Check if cycle_id exists, if not create index
    if 'cycle_id' not in df_pred.columns:
        df_pred['cycle_id'] = range(len(df_pred))
    
    df_sim = df_pred[sim_cols].copy()
    
    # Save
    df_sim.to_csv(output_path, index=False)
    print(f"\n✓ Simulation-ready predictions saved to: {output_path}")
    print(f"  Columns: {list(df_sim.columns)}")
    print(f"  Rows: {len(df_sim)}")
    
    return df_sim

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
        # Simulation-ready output
        predict_for_simulation(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output or "predictions_sim.csv",
            target_type=args.target_type,
            target_window=args.target_window,
            rolling_window=args.rolling_window
        )
    else:
        # Full prediction output
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