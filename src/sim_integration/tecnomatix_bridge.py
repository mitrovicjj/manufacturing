# src/sim_integration/tecnomatix_bridge.py
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ml.predict_production import ProductionPredictor

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    filename='logs/predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TecnomatixBridge:
    """Interface layer između Tecnomatix Plant Simulation i ML modela."""
    
    def __init__(self, model_path="models/production/PRODUCTION_MODEL.pkl"):
        self.predictor = ProductionPredictor(model_path=model_path)
        self.history_buffer = []  # ← NOVO: Čuva poslednjih N ciklusa
        self.max_history = 25     # ← Buffer size
        logging.info("TecnomatixBridge initialized successfully")
    
    def predict_json(self, json_input: str) -> str:
        """Interfejs za Tecnomatix - prima i vraća JSON string."""
        try:
            cycle_data = json.loads(json_input)
            result = self.predictor.predict_cycle(cycle_data)
            
            logging.info(
                f"Prediction - ID:{cycle_data.get('cycle_id', 'N/A')} "
                f"Risk:{result['risk_score']:.4f} "
                f"Trigger:{result['maintenance_trigger']} "
                f"Priority:{result['priority']}"
            )
            
            return json.dumps(result)
            
        except Exception as e:
            error_msg = f"Error in prediction: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return json.dumps({"error": str(e), "success": False})
    
    def predict_with_buffer(self, cycle_data: dict) -> dict:
        """
        NOVO: Automatski koristi istoriju ako je dostupna.
        Održava rolling buffer od poslednjih max_history ciklusa.
        """
        try:
            # Ako ima dovoljno istorije, koristi predict_with_history
            if len(self.history_buffer) >= 15:  # Min 15 ciklusa
                result = self.predictor.predict_with_history(
                    cycle_data, 
                    self.history_buffer
                )
                result['used_history'] = True
            else:
                # Fallback na predict_cycle
                result = self.predictor.predict_cycle(cycle_data)
                result['used_history'] = False
                result['buffer_size'] = len(self.history_buffer)
            
            # Dodaj trenutni ciklus u buffer
            self.history_buffer.append(cycle_data)
            
            # Održavaj buffer na max_history veličini
            if len(self.history_buffer) > self.max_history:
                self.history_buffer.pop(0)  # Ukloni najstariji
            
            # Logging
            logging.info(
                f"BufferPrediction - ID:{cycle_data.get('cycle_id', 'N/A')} "
                f"Risk:{result['risk_score']:.4f} "
                f"Trigger:{result['maintenance_trigger']} "
                f"Priority:{result['priority']} "
                f"UsedHistory:{result.get('used_history', False)} "
                f"BufferSize:{len(self.history_buffer)}"
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Error in predict_with_buffer: {str(e)}", exc_info=True)
            return {"error": str(e), "success": False}
    
    def clear_history(self):
        """Očisti buffer istorije (npr. nakon maintenance-a)"""
        buffer_size = len(self.history_buffer)
        self.history_buffer = []
        logging.info(f"History buffer cleared (was {buffer_size} cycles)")
        return {"success": True, "cleared_cycles": buffer_size}
    
    def get_buffer_status(self):
        """Vrati status buffer-a"""
        return {
            "buffer_size": len(self.history_buffer),
            "max_size": self.max_history,
            "fill_percentage": (len(self.history_buffer) / self.max_history) * 100
        }
    
    def should_trigger_maintenance(self, json_input: str) -> bool:
        """Boolean output za Tecnomatix decision logic."""
        try:
            result_json = self.predict_json(json_input)
            result = json.loads(result_json)
            
            if "error" in result:
                logging.error(f"Error in maintenance decision: {result['error']}")
                return False
            
            return result.get("maintenance_trigger", False)
            
        except Exception as e:
            logging.error(f"Exception in should_trigger_maintenance: {str(e)}")
            return False
    
    def get_risk_level(self, json_input: str) -> str:
        """Vraća priority level (LOW/MEDIUM/HIGH)."""
        try:
            result_json = self.predict_json(json_input)
            result = json.loads(result_json)
            
            if "error" in result:
                return "ERROR"
            
            return result.get("priority", "UNKNOWN")
        except Exception as e:
            logging.error(f"Exception in get_risk_level: {str(e)}")
            return "ERROR"
    
    def predict_json(self, json_input: str) -> str:
        """
        Interfejs za Tecnomatix - prima i vraća JSON string.
        """
        try:
            cycle_data = json.loads(json_input)
            
            # Predikcija
            result = self.predictor.predict_cycle(cycle_data)
            
            # Logging (sada result ima sve potrebne vrednosti)
            logging.info(
                f"Prediction - ID:{cycle_data.get('cycle_id', 'N/A')} "
                f"Risk:{result['risk_score']:.4f} "
                f"Trigger:{result['maintenance_trigger']} "
                f"Priority:{result['priority']}"
            )
            
            # ✅ Sada će raditi jer su sve vrednosti Python native tipovi
            return json.dumps(result)
            
        except Exception as e:
            error_msg = f"Error in prediction: {str(e)}"
            logging.error(error_msg, exc_info=True)  # ← Dodaj stack trace
            return json.dumps({"error": str(e), "success": False})
    
    def should_trigger_maintenance(self, json_input: str) -> bool:
        """
        Boolean output za Tecnomatix decision logic.
        """
        try:
            result_json = self.predict_json(json_input)
            result = json.loads(result_json)
            
            if "error" in result:
                logging.error(f"Error in maintenance decision: {result['error']}")
                return False
            
            return result.get("maintenance_trigger", False)
            
        except Exception as e:
            logging.error(f"Exception in should_trigger_maintenance: {str(e)}")
            return False
    
    def get_risk_level(self, json_input: str) -> str:
        """
        Vraća priority level (LOW/MEDIUM/HIGH).
        """
        try:
            result_json = self.predict_json(json_input)
            result = json.loads(result_json)
            
            if "error" in result:
                return "ERROR"
            
            return result.get("priority", "UNKNOWN")
        except Exception as e:
            logging.error(f"Exception in get_risk_level: {str(e)}")
            return "ERROR"


# --- TESTING ---
if __name__ == "__main__":
    print("="*70)
    print("🔧 TECNOMATIX BRIDGE - TESTING SUITE")
    print("="*70)
    
    bridge = TecnomatixBridge()
    
    # Test 1: Normal operation
    print("\n📊 TEST 1: Normal Operation")
    print("-" * 70)
    normal_cycle = {
        "cycle_id": 1001,
        "cycle_time": 60,
        "temperature": 30,
        "vibration": 0.02,
        "pressure": 5.0,
        "operator": "op1",
        "maintenance_type": "preventive"
    }
    normal_json = json.dumps(normal_cycle)
    
    result1 = bridge.predict_json(normal_json)
    result1_dict = json.loads(result1)
    print(f"Input: {normal_json}")
    print(f"Output: {result1}")
    print(f"Maintenance Trigger: {bridge.should_trigger_maintenance(normal_json)}")
    print(f"Risk Level: {bridge.get_risk_level(normal_json)}")
    
    # Test 2: Degraded operation
    print("\n⚠️  TEST 2: Degraded Operation")
    print("-" * 70)
    degraded_cycle = {
        "cycle_id": 1002,
        "cycle_time": 75,
        "temperature": 45,
        "vibration": 0.08,
        "pressure": 4.2,
        "operator": "op2",
        "maintenance_type": "corrective"
    }
    degraded_json = json.dumps(degraded_cycle)
    
    result2 = bridge.predict_json(degraded_json)
    result2_dict = json.loads(result2)
    print(f"Input: {degraded_json}")
    print(f"Output: {result2}")
    print(f"Maintenance Trigger: {bridge.should_trigger_maintenance(degraded_json)}")
    print(f"Risk Level: {bridge.get_risk_level(degraded_json)}")
    
    # Test 3: Critical conditions
    print("\n🚨 TEST 3: Critical Conditions")
    print("-" * 70)
    critical_cycle = {
        "cycle_id": 1003,
        "cycle_time": 90,
        "temperature": 55,
        "vibration": 0.15,
        "pressure": 3.5,
        "operator": "op3",
        "maintenance_type": "emergency"
    }
    critical_json = json.dumps(critical_cycle)
    
    result3 = bridge.predict_json(critical_json)
    result3_dict = json.loads(result3)
    print(f"Input: {critical_json}")
    print(f"Output: {result3}")
    print(f"Maintenance Trigger: {bridge.should_trigger_maintenance(critical_json)}")
    print(f"Risk Level: {bridge.get_risk_level(critical_json)}")
    
    # Test 4: Invalid input
    print("\n❌ TEST 4: Invalid Input Handling")
    print("-" * 70)
    invalid_json = '{"invalid": "data"}'
    result4 = bridge.predict_json(invalid_json)
    print(f"Input: {invalid_json}")
    print(f"Output: {result4}")
    
    # Test 5: Batch processing - SA ERROR HANDLING
    print("\n📈 TEST 5: Batch Processing")
    print("-" * 70)
    for i in range(5):
        cycle = {
            "cycle_id": 2000 + i,
            "cycle_time": 60 + i * 2,
            "temperature": 30 + i * 1.5,
            "vibration": 0.02 + i * 0.005,
            "pressure": 5.0 - i * 0.1,
            "operator": f"op{i % 3 + 1}",
            "maintenance_type": "preventive"
        }
        
        result_json = bridge.predict_json(json.dumps(cycle))
        result = json.loads(result_json)
        
        # ✅ Proveri da li ima error pre pristupa
        if "error" in result:
            print(f"Cycle {cycle['cycle_id']}: ERROR - {result['error']}")
        else:
            print(f"Cycle {cycle['cycle_id']}: Risk={result['risk_score']:.4f}, "
                  f"Priority={result['priority']}")
    
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    print(f"Check logs/predictions.log for detailed logging")
    print("="*70)