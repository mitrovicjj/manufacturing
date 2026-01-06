"""
ANFIS Configuration File
========================
Centralizovana konfiguracija za ANFIS model.
"""

import numpy as np
from typing import Dict, List, Tuple


class ANFISConfig:
    """Glavna konfiguracija ANFIS modela."""
    
    def __init__(self, n_inputs=None, n_mfs_per_input=None, feature_ranges=None, 
                 use_domain_knowledge=True):
        """
        Initialize ANFIS config sa mogućnošću dynamic override-a.
        
        Args:
            n_inputs: Broj input feature-a (default: 4)
            n_mfs_per_input: Lista MF-ova po input-u (default: [3, 3, 2, 2])
            feature_ranges: Lista (min, max) tuple-ova (default: predefined)
            use_domain_knowledge: Da li koristiti domain knowledge (default: True)
        """
        # MODEL ARCHITECTURE
        self.N_INPUTS = n_inputs if n_inputs is not None else 4
        self.N_MFS_PER_INPUT = n_mfs_per_input if n_mfs_per_input is not None else [3, 3, 2, 2]
        self.MF_TYPE = 'gaussian'
        self.USE_DOMAIN_KNOWLEDGE = use_domain_knowledge
        
        # FEATURE CONFIGURATION
        if self.N_INPUTS == 4:
            self.FEATURE_NAMES = ['Temperature', 'Vibration', 'Cycle Time', 'Tool Wear']
        else:
            self.FEATURE_NAMES = [f'Feature_{i}' for i in range(self.N_INPUTS)]
        
        # Feature ranges
        if feature_ranges is not None:
            self.FEATURE_RANGES = feature_ranges
        elif self.N_INPUTS == 4:
            self.FEATURE_RANGES = [
                (15.0, 35.0),   # Temperature
                (0.01, 0.10),   # Vibration
                (40.0, 90.0),   # Cycle Time
                (1.0, 2.0)      # Tool Wear
            ]
        else:
            self.FEATURE_RANGES = [(0.0, 1.0)] * self.N_INPUTS
        
        # DOMAIN KNOWLEDGE
        if self.N_INPUTS == 4 and use_domain_knowledge:
            self.DOMAIN_KNOWLEDGE = [
                {
                    'name': 'Temperature',
                    'unit': '°C',
                    'safe_max': 24.0,
                    'warning': 26.0,
                    'critical': 30.0,
                    'standard': 'OSHA + Dataset Analysis',
                },
                {
                    'name': 'Vibration',
                    'unit': 'mm/s',
                    'safe_max': 0.025,
                    'warning': 0.028,
                    'critical': 0.045,
                    'standard': 'ISO 10816',
                },
                {
                    'name': 'Cycle Time',
                    'unit': 's',
                    'safe_max': 65.0,
                    'warning': 70.0,
                    'critical': 80.0,
                    'standard': 'Process Analysis',
                },
                {
                    'name': 'Tool Wear',
                    'unit': 'factor',
                    'safe_max': 1.3,
                    'warning': 1.5,
                    'critical': 1.7,
                    'standard': 'Predictive Maintenance',
                }
            ]
        else:
            self.DOMAIN_KNOWLEDGE = {}
        
        # LINGUISTIC TERMS
        max_mfs = max(self.N_MFS_PER_INPUT) if self.N_MFS_PER_INPUT else 2
        if max_mfs == 2:
            self.LINGUISTIC_TERMS = ['LOW', 'HIGH']
        elif max_mfs == 3:
            self.LINGUISTIC_TERMS = ['LOW', 'MEDIUM', 'HIGH']
        else:
            self.LINGUISTIC_TERMS = [f'MF_{i}' for i in range(max_mfs)]
        
        # TRAINING HYPERPARAMETERS
        self.LEARNING_RATE = 0.01
        self.BATCH_SIZE = 64
        self.EPOCHS = 100
        self.EARLY_STOPPING_PATIENCE = 15
        self.MIN_DELTA = 1e-6
        self.WEIGHT_DECAY = 0.0001
        
        # RISK THRESHOLDS
        self.RISK_THRESHOLDS = {
            'low': 0.35,
            'high': 0.60
        }
        
        self.RISK_LABELS = {
            'low': '✅ LOW RISK',
            'medium': '⚠️  MEDIUM RISK',
            'high': '🚨 HIGH RISK'
        }
        
        # VISUALIZATION & PERSISTENCE
        self.VIZ_COLORS = {
            'low': '#2E86AB',
            'medium': '#F77F00',
            'high': '#06A77D'
        }
        self.VIZ_DPI = 300
        self.VIZ_OUTPUT_DIR = 'anfis_plots'
        self.MODEL_SAVE_PATH = 'models/anfis_trained.pkl'
        self.CHECKPOINT_DIR = 'checkpoints'
        self.LOG_LEVEL = 'INFO'
        self.LOG_FILE = 'anfis_training.log'
        self.VERBOSE_TRAINING = True
        self.RANDOM_SEED = 42
    
    # HELPER METHODS
    def get_risk_level(self, output_value: float) -> str:
        """Mapira ANFIS output u risk level."""
        if output_value < self.RISK_THRESHOLDS['low']:
            return 'low'
        elif output_value > self.RISK_THRESHOLDS['high']:
            return 'high'
        else:
            return 'medium'
    
    def get_risk_label(self, output_value: float) -> str:
        """Vraća emoji label za risk level."""
        level = self.get_risk_level(output_value)
        return self.RISK_LABELS[level]
    
    def validate_config(self) -> bool:
        """Validira konfiguraciju."""
        if len(self.FEATURE_RANGES) != self.N_INPUTS:
            raise ValueError(f"FEATURE_RANGES length mismatch: {len(self.FEATURE_RANGES)} != {self.N_INPUTS}")
        
        if len(self.N_MFS_PER_INPUT) != self.N_INPUTS:
            raise ValueError(f"N_MFS_PER_INPUT length mismatch: {len(self.N_MFS_PER_INPUT)} != {self.N_INPUTS}")
        
        if self.RISK_THRESHOLDS['low'] >= self.RISK_THRESHOLDS['high']:
            raise ValueError("RISK_THRESHOLDS['low'] must be < ['high']")
        
        print("✅ Config validation passed!")
        return True
    
    def print_config(self):
        """Pretty print konfiguracije."""
        print("\n" + "="*70)
        print("⚙️  ANFIS CONFIGURATION")
        print("="*70)
        print(f"\n📊 Model Architecture:")
        print(f"   Inputs: {self.N_INPUTS}")
        print(f"   MFs per input: {self.N_MFS_PER_INPUT}")
        print(f"   Total rules: {np.prod(self.N_MFS_PER_INPUT)}")
        print(f"   MF type: {self.MF_TYPE}")
        print(f"\n🎓 Training:")
        print(f"   Epochs: {self.EPOCHS}")
        print(f"   Learning rate: {self.LEARNING_RATE}")
        print(f"   Batch size: {self.BATCH_SIZE}")
        print("="*70 + "\n")


# USAGE EXAMPLE
if __name__ == "__main__":
    # Test default config
    config = ANFISConfig()
    config.validate_config()
    config.print_config()
    
    # Test dynamic config
    print("\nTesting dynamic config (8 inputs):")
    config_dynamic = ANFISConfig(
        n_inputs=8,
        n_mfs_per_input=[2]*8,
        feature_ranges=[(0, 1)]*8,
        use_domain_knowledge=False
    )
    config_dynamic.print_config()
    print(f"✅ Dynamic config test passed!")
