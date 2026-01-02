import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.predict import predict_final_for_simulation

class InferenceConfig:
    """
    Configuration for running final prediction / inference.
    """
    def __init__(self, experiment_name, base_dir="C:/Users/Korisnik/py/manufacturing"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_dir, "experiments", f"{experiment_name}_{self.timestamp}")
        self.predictions_path = os.path.join(self.experiment_dir, "predictions.csv")
        os.makedirs(self.experiment_dir, exist_ok=True)

    def print_config(self, data_path):
        print("\n" + "="*70)
        print(f"INFERENCE EXPERIMENT: {self.experiment_name}")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Experiment dir: {self.experiment_dir}")
        print(f"Data for prediction: {data_path} (raw CSV, feature engineering will run automatically)")
        print("="*70 + "\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run final model inference")
    
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name of experiment (results saved to experiments/{name}_{timestamp}/)")
    parser.add_argument("--base_dir", type=str, 
                        default="C:/Users/Korisnik/py/manufacturing",
                        help="Base directory of project")
    parser.add_argument("--predict_final", action="store_true",
                        help="Run prediction using final model for Tecnomatix simulation")
    parser.add_argument("--predict_data_path", type=str, default=None,
                        help="Path to raw CSV for final prediction (feature engineering will run automatically)")

    args = parser.parse_args()

    if args.predict_final:
        # default: use raw data
        data_path = args.predict_data_path or "data/raw/logs_all_machines.csv"
        config = InferenceConfig(experiment_name=args.experiment_name, base_dir=args.base_dir)
        config.print_config(data_path)
        
        print("\n" + "🔮 "*20)
        print("STEP: FINAL PREDICTION / INFERENCE")
        print("🔮 "*20 + "\n")
        
        # pipeline will run feature engineering automatically
        predict_final_for_simulation(data_path=data_path, output_path=config.predictions_path)
        
        print("\n" + "="*70)
        print("✅ INFERENCE COMPLETE")
        print(f"Predictions saved to: {config.predictions_path}")
        print("="*70 + "\n")
        sys.exit(0)

if __name__ == "__main__":
    main()