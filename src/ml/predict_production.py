import joblib
import pandas as pd
import numpy as np  # ← Dodaj numpy import
import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ml.feature_store import FeatureStore
from src.ml.features import build_inference_features


class ProductionPredictor:
    def __init__(self, model_path="models/production/PRODUCTION_MODEL.pkl"):
        """
        Prediktor za Tecnomatix Plant Simulation.
        """
        self.model = joblib.load(model_path)
        self.feature_store = FeatureStore(cache_dir="data/feature_cache/")
        
        if hasattr(self.model, 'feature_names_in_'):
            self.model_features = self.model.feature_names_in_
            print(f"Model očekuje {len(self.model_features)} features")
        else:
            self.model_features = None


    def predict_cycle(self, cycle_data: dict):
        """
        Predikcija za pojedinačni ciklus.
        
        Args:
            cycle_data: dict sa raw kolonama
        
        Returns:
            dict: {'risk_score': float, 'maintenance_trigger': bool, 'priority': str}
        """
        df_cycle = pd.DataFrame([cycle_data])
        
        config = {
            "cycle_id": cycle_data.get("cycle_id", 0),
            "inference": True
        }
        
        df_features = self.feature_store.get_features(config)
        
        if df_features is None:
            df_features = build_inference_features(df_cycle)
            
            if self.model_features is not None:
                missing_cols = set(self.model_features) - set(df_features.columns)
                for col in missing_cols:
                    df_features[col] = 0
                    print(f"⚠️  Added missing column '{col}' with default value 0")
                
                df_features = df_features[self.model_features]
            
            self.feature_store.save_features(df_features, config)
        
        # Predikcija
        risk_score = self.model.predict_proba(df_features)[0, 1]
        maintenance_trigger = risk_score >= 0.35
        
        priority = "HIGH" if risk_score > 0.6 else "MEDIUM" if risk_score > 0.35 else "LOW"
        
        # ✅ KLJUČNA IZMENA: Konvertuj sve u Python native tipove
        return {
            "risk_score": float(risk_score),              # numpy.float64 → float
            "maintenance_trigger": bool(maintenance_trigger),  # numpy.bool_ → bool
            "priority": str(priority)                      # Osiguraj da je string
        }
    def predict_with_history(self, current_cycle: dict, history: list):
        """
        Predikcija sa istorijom - PREPORUČENO za produkciju.
        
        Args:
            current_cycle: Trenutni ciklus
            history: Lista prethodnih ciklusa (min 20 za rolling window)
        
        Returns:
            dict: Predikcija sa realističnijim rolling/lag feature-ima
        """
        # Kombinuj istoriju + trenutni ciklus
        all_cycles = history + [current_cycle]
        df = pd.DataFrame(all_cycles)
        
        # Generiši feature-e sa pravom istorijom
        df_features = build_inference_features(df)
        
        if self.model_features is not None:
            missing_cols = set(self.model_features) - set(df_features.columns)
            for col in missing_cols:
                df_features[col] = 0
            df_features = df_features[self.model_features]
        
        # Predikcija samo za poslednji red (trenutni ciklus)
        risk_score = self.model.predict_proba(df_features.iloc[[-1]])[0, 1]
        maintenance_trigger = risk_score >= 0.35
        priority = "HIGH" if risk_score > 0.6 else "MEDIUM" if risk_score > 0.35 else "LOW"
        
        return {
            "risk_score": float(risk_score),
            "maintenance_trigger": bool(maintenance_trigger),
            "priority": str(priority),
            "history_length": len(history)
        }

# --- TEST ---
if __name__ == "__main__":
    # Obriši stari cache
    print("🗑️  Čistim stari cache...")
    shutil.rmtree("data/feature_cache", ignore_errors=True)
    os.makedirs("data/feature_cache", exist_ok=True)
    
    # Test cycle
    test_cycle = {
        "cycle_id": 123,
        "cycle_time": 60,
        "temperature": 30,
        "vibration": 0.02,
        "pressure": 5.0,
        "operator": "op1",
        "maintenance_type": "preventive"
    }

    predictor = ProductionPredictor()
    result = predictor.predict_cycle(test_cycle)
    
    # Test JSON serializacije
    import json
    try:
        json_result = json.dumps(result)
        print("\n✅ JSON serialization successful!")
        print(json_result)
    except TypeError as e:
        print(f"\n❌ JSON serialization failed: {e}")
    
    print("\n" + "="*50)
    print("📊 PREDIKCIJA ZA TEST CYCLE:")
    print("="*50)
    print(f"  Risk Score: {result['risk_score']:.4f}")
    print(f"  Maintenance Trigger: {result['maintenance_trigger']}")
    print(f"  Priority: {result['priority']}")
    print("="*50)