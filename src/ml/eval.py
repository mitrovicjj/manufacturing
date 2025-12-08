import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

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


# EVALUATION

def evaluate(model, X, y, output_dir):
    """
    Evaluate model and generate metrics + plots.
    
    Args:
        model: Trained model
        X: Test features
        y: Test labels
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate predictions
    probs, preds = predict(model, X)

    # Compute metrics
    roc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    f1 = f1_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    cm = confusion_matrix(y, preds).tolist()

    results = dict(
        roc_auc=float(roc),
        pr_auc=float(pr_auc),
        f1=float(f1),
        precision=float(prec),
        recall=float(rec),
        confusion_matrix=cm,
        true_positive_rate=float((y == 1).sum()),
        predicted_positive_rate=float(preds.sum()),
        total_samples=int(len(y))
    )

    # Save metrics to JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"ROC-AUC:         {roc:.4f}")
    print(f"PR-AUC:          {pr_auc:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
    print("="*70)

    # --------------------------
    # PLOTS
    # --------------------------

    # 1. Histogram of predicted probabilities
    plt.figure(figsize=(8, 5))
    plt.hist(probs[y == 0], bins=30, alpha=0.6, label='Negative class', color='blue')
    plt.hist(probs[y == 1], bins=30, alpha=0.6, label='Positive class', color='red')
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Prediction Distribution by True Class", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "probs_hist.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recall_vals, precision_vals, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Risk score distribution
    plt.figure(figsize=(8, 5))
    risk_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    risk_labels = ['Low\n(0-0.3)', 'Medium-Low\n(0.3-0.5)', 'Medium\n(0.5-0.7)', 
                   'Medium-High\n(0.7-0.9)', 'High\n(0.9-1.0)']
    risk_categories = pd.cut(probs, bins=risk_bins, labels=risk_labels, include_lowest=True)
    risk_counts = risk_categories.value_counts().sort_index()
    
    plt.bar(range(len(risk_counts)), risk_counts.values, color=['green', 'yellowgreen', 'yellow', 'orange', 'red'])
    plt.xticks(range(len(risk_counts)), risk_counts.index, fontsize=10)
    plt.ylabel("Count", fontsize=12)
    plt.title("Risk Score Distribution", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "risk_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Metrics saved to: {output_dir}/metrics.json")
    print(f"✓ Plots saved to: {output_dir}/")

    return results

# EVALUATE ON TEST DATA

def evaluate_on_data(model_path,
                     test_csv,
                     output_dir,
                     target_type='windowed',
                     target_window=5,
                     rolling_window=20):
    """
    Complete evaluation pipeline on test data.
    
    Args:
        model_path: Path to trained model (.pkl)
        test_csv: Path to raw test CSV
        output_dir: Directory to save evaluation results
        target_type: 'next' or 'windowed' (must match training!)
        target_window: Window size if windowed
        rolling_window: Rolling window size (must match training!)
    """
    print("="*70)
    print("EVALUATION PIPELINE START")
    print("="*70)

    # 1. Load model
    print(f"\n[1/4] Loading model from: {model_path}")
    model = load_model(model_path)
    print("  ✓ Model loaded")

    # 2. Load test data
    print(f"\n[2/4] Loading test data from: {test_csv}")
    df = pd.read_csv(test_csv)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # 3. Feature engineering (MUST MATCH TRAINING!)
    print(f"\n[3/4] Feature engineering...")
    print(f"  Config: target_type={target_type}, target_window={target_window}, rolling_window={rolling_window}")
    
    df_features = build_features(
        df,
        target_type=target_type,
        target_window=target_window,
        rolling_window=rolling_window
    )

    # Prepare X and y
    target_col = 'downtime_next' if target_type == 'next' else f'downtime_next_{target_window}'
    
    feature_cols = get_feature_columns(df_features, target_col=target_col)
    X = df_features[feature_cols]
    y = df_features[target_col]

    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Target: {target_col}")
    print(f"  Test samples: {len(X)}")

    # 4. Evaluate
    print(f"\n[4/4] Running evaluation...")
    evaluate(model, X, y, output_dir)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


# CLI

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictive maintenance model")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pkl)")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to raw test CSV data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    
    # Optional arguments (must match training config!)
    parser.add_argument("--target_type", type=str, default='windowed',
                        choices=['next', 'windowed'],
                        help="Target type (must match training)")
    parser.add_argument("--target_window", type=int, default=5,
                        help="Target window (must match training)")
    parser.add_argument("--rolling_window", type=int, default=20,
                        help="Rolling window (must match training)")

    args = parser.parse_args()

    evaluate_on_data(
        model_path=args.model,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        target_type=args.target_type,
        target_window=args.target_window,
        rolling_window=args.rolling_window
    )


if __name__ == "__main__":
    main()