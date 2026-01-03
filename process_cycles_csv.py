# process_cycles_csv_final.py
"""
FINALNI PRODUCTION-READY PIPELINE:
- Koristi feature CSV (rolling_window=30)
- Odbacuje warm-up period (prvih 30 ciklusa)
- Output za Power BI sa čistim nazivom
"""

import pandas as pd
import sys
import os
import joblib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ml.features import get_feature_columns
from config import FINAL_MODEL_PATH, FINAL_THRESHOLD

# ===============================
# PUTANJE
# ===============================
FEATURES_CSV = r"C:\Users\Korisnik\py\manufacturing\data\processed\logs_ml_features_rw30.csv"
OUTPUT_CSV = r"C:\Users\Korisnik\py\manufacturing\data\processed\cycles_decisions_powerbi.csv"

WARMUP_PERIOD = 30  # Broj redova za odbacivanje na početku


def process_cycles():
    print("=" * 70)
    print("📁 FINALNI ML PIPELINE ZA POWER BI")
    print("=" * 70)

    # 1. Učitaj feature CSV
    try:
        df_features = pd.read_csv(FEATURES_CSV)
        print(f"\n✅ Loaded feature CSV: {len(df_features)} cycles")
        print(f"   Path: {FEATURES_CSV}")
    except Exception as e:
        print(f"\n❌ ERROR: Feature CSV ne postoji!")
        print(f"   Prvo pokreni: python generate_features.py")
        return 1

    # 2. Load ML model
    try:
        print("\n🔄 Loading ML model...")
        model = joblib.load(FINAL_MODEL_PATH)
        if hasattr(model, 'n_jobs'):
            model.n_jobs = 1
        print(f"✅ Model loaded")
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        return 1

    # 3. Pripremi feature-eve
    print("\n📊 Preparing features for prediction...")
    
    target_col = 'downtime_next_5'
    feature_cols = get_feature_columns(df_features, target_col=target_col)
    X = df_features[feature_cols]
    
    print(f"   Features: {len(feature_cols)} columns")

    # 4. Batch prediction
    print(f"\n🔄 Running batch predictions...")
    try:
        risk_scores = model.predict_proba(X)[:, 1]
        maintenance_trigger = risk_scores >= FINAL_THRESHOLD
        
        priorities = []
        for score in risk_scores:
            if score > 0.6:
                priorities.append("HIGH")
            elif score > FINAL_THRESHOLD:
                priorities.append("MEDIUM")
            else:
                priorities.append("LOW")
        
        print(f"✅ Predictions complete!")
        
    except Exception as e:
        print(f"\n❌ ERROR during prediction: {e}")
        return 1

    # 5. Create output DataFrame
    df_output = pd.DataFrame({
        "timestamp": df_features["timestamp"],
        "machine_id": df_features["machine_id"],
        "cycle_id": df_features["cycle_id"],
        "cycle_time": df_features["cycle_time"],
        "temperature": df_features["temperature"],
        "vibration": df_features["vibration"],
        "pressure": df_features["pressure"],
        "risk_score": risk_scores,
        "maintenance_trigger": maintenance_trigger,
        "priority": priorities
    })

    # 6. ODBACI WARM-UP PERIOD (prvih 30 redova)
    print(f"\n⚠️  Removing warm-up period (first {WARMUP_PERIOD} cycles)...")
    print(f"   Before: {len(df_output)} cycles")
    df_output = df_output.iloc[WARMUP_PERIOD:]
    df_output = df_output.reset_index(drop=True)
    print(f"   After:  {len(df_output)} cycles")

    # 7. Save output
    try:
        df_output.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Output CSV saved: {OUTPUT_CSV}")
        print(f"   Rows: {len(df_output)}")
    except Exception as e:
        print(f"\n❌ ERROR saving output: {e}")
        return 1

    # 8. Summary
    print("\n📊 SUMMARY (bez warm-up perioda)")
    print(f"   Threshold: {FINAL_THRESHOLD}")
    print(f"   LOW:    {priorities[WARMUP_PERIOD:].count('LOW')} ({priorities[WARMUP_PERIOD:].count('LOW')/len(df_output)*100:.1f}%)")
    print(f"   MEDIUM: {priorities[WARMUP_PERIOD:].count('MEDIUM')} ({priorities[WARMUP_PERIOD:].count('MEDIUM')/len(df_output)*100:.1f}%)")
    print(f"   HIGH:   {priorities[WARMUP_PERIOD:].count('HIGH')} ({priorities[WARMUP_PERIOD:].count('HIGH')/len(df_output)*100:.1f}%)")
    print(f"   Maintenance triggers: {df_output['maintenance_trigger'].sum()} ({df_output['maintenance_trigger'].mean()*100:.1f}%)")
    print(f"   Avg risk score: {df_output['risk_score'].mean():.4f}")
    print(f"   Max risk score: {df_output['risk_score'].max():.4f}")

    # 9. Sample output
    print("\n📋 Sample output (first 5 rows AFTER warm-up):")
    print(df_output.head().to_string(index=False))

    print("\n" + "=" * 70)
    print("✅ FINALNI PIPELINE COMPLETE - SPREMAN ZA POWER BI!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(process_cycles())