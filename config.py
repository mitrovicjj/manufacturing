from datetime import datetime
import os

class ExperimentConfig:
    def __init__(self, experiment_name, base_dir="C:/Users/Korisnik/py/manufacturing"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        
        self.raw_data_path = os.path.join(base_dir, "data/raw/logs_all_machines.csv")
        self.experiment_dir = os.path.join(base_dir, "experiments", f"{experiment_name}_{self.timestamp}")
        self.model_path = os.path.join(self.experiment_dir, "model.pkl")
        self.features_path = os.path.join(self.experiment_dir, "features.csv")
        self.predictions_path = os.path.join(self.experiment_dir, "predictions.csv")
        self.eval_dir = os.path.join(self.experiment_dir, "evaluation")

    def create_dirs(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

BASE_DIR = "C:/Users/Korisnik/py/manufacturing"

FINAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgb_pm_final.pkl"
)

FINAL_THRESHOLD = 0.35
FINAL_PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "predictions_pm_final.csv"
)