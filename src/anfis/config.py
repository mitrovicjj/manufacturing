"""
ANFIS Configuration File
========================
Centralizovana konfiguracija za ANFIS model.
Sve konstante, threshold-e i hyperparametre držimo ovde.
"""

import numpy as np
from typing import Dict, List, Tuple


class ANFISConfig:
    """
    Glavna konfiguracija ANFIS modela.
    """
    
    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    
    N_INPUTS: int = 4
    """Broj input feature-a"""
    
    N_MFS_PER_INPUT: List[int] = [3, 3, 2, 2]
    """Broj membership functions po input-u (3×3×2×2 = 36 pravila)"""
    
    MF_TYPE: str = 'gaussian'
    """Tip membership function: 'gaussian', 'bell', 'trapezoid'"""
    
    USE_DOMAIN_KNOWLEDGE: bool = True
    """Da li koristiti domain-aware inicijalizaciju"""
    
    # ==========================================
    # FEATURE CONFIGURATION
    # ==========================================
    
    FEATURE_NAMES: List[str] = [
        'Temperature',
        'Vibration', 
        'Cycle Time',
        'Tool Wear'
    ]
    """Nazivi input feature-a"""
    
    FEATURE_RANGES: List[Tuple[float, float]] = [
        (15.0, 35.0),   # Temperature (°C)
        (0.01, 0.10),   # Vibration (mm/s)
        (40.0, 90.0),   # Cycle Time (s)
        (1.0, 2.0)      # Tool Wear (factor)
    ]
    """Min/max range za svaki feature"""
    
    # ==========================================
    # DOMAIN KNOWLEDGE - Industrial Thresholds
    # ==========================================
    
    DOMAIN_KNOWLEDGE: List[Dict] = [
        {
            'name': 'Temperature',
            'unit': '°C',
            'safe_max': 24.0,      # Optimalna temperatura
            'warning': 26.0,       # Prelazak u degraded state
            'critical': 30.0,      # High risk zona
            'standard': 'OSHA + Dataset Analysis',
            'description': 'Operating temperature of machinery'
        },
        {
            'name': 'Vibration',
            'unit': 'mm/s',
            'safe_max': 0.025,     # ISO 10816 Zone A
            'warning': 0.028,      # Zone B - acceptable
            'critical': 0.045,     # Zone C - unsatisfactory
            'standard': 'ISO 10816',
            'description': 'Vibration velocity monitoring'
        },
        {
            'name': 'Cycle Time',
            'unit': 's',
            'safe_max': 65.0,      # Nominalni cycle time
            'warning': 70.0,       # Sporiji od normalnog
            'critical': 80.0,      # Performance issue
            'standard': 'Process Analysis',
            'description': 'Manufacturing cycle duration'
        },
        {
            'name': 'Tool Wear',
            'unit': 'factor',
            'safe_max': 1.3,       # Dobar alat
            'warning': 1.5,        # Maintenance needed
            'critical': 1.7,       # Replacement urgency
            'standard': 'Predictive Maintenance',
            'description': 'Tool degradation factor'
        }
    ]
    """Domain knowledge sa industrijskim threshold-ima"""
    
    # ==========================================
    # LINGUISTIC TERMS
    # ==========================================
    
    LINGUISTIC_TERMS: List[str] = ['LOW', 'MEDIUM', 'HIGH']
    """Fuzzy lingvistički termini"""
    
    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    
    # Optimizer settings
    LEARNING_RATE: float = 0.01
    """Learning rate za Adam optimizer"""
    
    BATCH_SIZE: int = 64
    """Batch size za mini-batch gradient descent"""
    
    EPOCHS: int = 100
    """Maksimalan broj training epochs"""
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 15
    """Broj epoha bez poboljšanja pre early stopping-a"""
    
    MIN_DELTA: float = 1e-6
    """Minimalna promena loss-a da bi se smatralo poboljšanjem"""
    
    # Regularization
    WEIGHT_DECAY: float = 0.0001
    """L2 regularization strength"""
    
    # ==========================================
    # RISK ASSESSMENT THRESHOLDS
    # ==========================================
    
    RISK_THRESHOLDS: Dict[str, float] = {
        'low': 0.35,      # Output < 0.35 → LOW RISK
        'high': 0.60      # Output > 0.60 → HIGH RISK
    }
    """Threshold-i za kategorizaciju risk level-a"""
    
    RISK_LABELS: Dict[str, str] = {
        'low': '✅ LOW RISK',
        'medium': '⚠️  MEDIUM RISK',
        'high': '🚨 HIGH RISK'
    }
    """Emoji labele za risk level-e"""
    
    # ==========================================
    # VISUALIZATION SETTINGS
    # ==========================================
    
    VIZ_COLORS: Dict[str, str] = {
        'low': '#2E86AB',      # Plava
        'medium': '#F77F00',   # Narandžasta
        'high': '#06A77D'      # Zelena
    }
    """Boje za vizualizaciju MF-ova"""
    
    VIZ_DPI: int = 300
    """DPI za saved plots"""
    
    VIZ_OUTPUT_DIR: str = 'anfis_plots'
    """Folder za čuvanje vizualizacija"""
    
    # ==========================================
    # MODEL PERSISTENCE
    # ==========================================
    
    MODEL_SAVE_PATH: str = 'models/anfis_trained.pkl'
    """Default path za čuvanje modela"""
    
    CHECKPOINT_DIR: str = 'checkpoints'
    """Folder za training checkpoints"""
    
    # ==========================================
    # LOGGING
    # ==========================================
    
    LOG_LEVEL: str = 'INFO'
    """Logging level: DEBUG, INFO, WARNING, ERROR"""
    
    LOG_FILE: str = 'anfis_training.log'
    """Log file path"""
    
    VERBOSE_TRAINING: bool = True
    """Print detaljne training informacije"""
    
    # ==========================================
    # RANDOM SEED
    # ==========================================
    
    RANDOM_SEED: int = 42
    """Random seed za reproducibilnost"""
    
    # ==========================================
    # HELPER METHODS
    # ==========================================
    
    @classmethod
    def get_risk_level(cls, output_value: float) -> str:
        """
        Mapira ANFIS output u risk level.
        
        Args:
            output_value: ANFIS output [0, 1]
        
        Returns:
            Risk level string ('low', 'medium', 'high')
        """
        if output_value < cls.RISK_THRESHOLDS['low']:
            return 'low'
        elif output_value > cls.RISK_THRESHOLDS['high']:
            return 'high'
        else:
            return 'medium'
    
    @classmethod
    def get_risk_label(cls, output_value: float) -> str:
        """
        Vraća emoji label za risk level.
        
        Args:
            output_value: ANFIS output [0, 1]
        
        Returns:
            Emoji label string
        """
        level = cls.get_risk_level(output_value)
        return cls.RISK_LABELS[level]
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validira konfiguraciju za konzistentnost.
        
        Returns:
            True ako je config validan
        
        Raises:
            ValueError: Ako config nije validan
        """
        # Check broj feature-a
        if len(cls.FEATURE_RANGES) != cls.N_INPUTS:
            raise ValueError(f"FEATURE_RANGES dužina ({len(cls.FEATURE_RANGES)}) "
                           f"!= N_INPUTS ({cls.N_INPUTS})")
        
        if len(cls.DOMAIN_KNOWLEDGE) != cls.N_INPUTS:
            raise ValueError(f"DOMAIN_KNOWLEDGE dužina ({len(cls.DOMAIN_KNOWLEDGE)}) "
                           f"!= N_INPUTS ({cls.N_INPUTS})")
        
        # Check n_mfs_per_input
        if len(cls.N_MFS_PER_INPUT) != cls.N_INPUTS:
            raise ValueError(f"N_MFS_PER_INPUT dužina ({len(cls.N_MFS_PER_INPUT)}) "
                           f"!= N_INPUTS ({cls.N_INPUTS})")
        
        # Check risk thresholds
        if cls.RISK_THRESHOLDS['low'] >= cls.RISK_THRESHOLDS['high']:
            raise ValueError("RISK_THRESHOLDS['low'] mora biti < ['high']")
        
        print("✅ Config validation passed!")
        return True
    
    @classmethod
    def print_config(cls):
        """Pretty print konfiguracije."""
        print("\n" + "="*70)
        print("⚙️  ANFIS CONFIGURATION")
        print("="*70)
        print(f"\n📊 Model Architecture:")
        print(f"   Inputs: {cls.N_INPUTS}")
        print(f"   MFs per input: {cls.N_MFS_PER_INPUT}")
        print(f"   Total rules: {np.prod(cls.N_MFS_PER_INPUT)}")
        print(f"   MF type: {cls.MF_TYPE}")
        
        print(f"\n🎓 Training:")
        print(f"   Epochs: {cls.EPOCHS}")
        print(f"   Learning rate: {cls.LEARNING_RATE}")
        print(f"   Batch size: {cls.BATCH_SIZE}")
        print(f"   Early stopping patience: {cls.EARLY_STOPPING_PATIENCE}")
        
        print(f"\n🎯 Risk Thresholds:")
        print(f"   LOW < {cls.RISK_THRESHOLDS['low']}")
        print(f"   MEDIUM: {cls.RISK_THRESHOLDS['low']} - {cls.RISK_THRESHOLDS['high']}")
        print(f"   HIGH > {cls.RISK_THRESHOLDS['high']}")
        
        print("="*70 + "\n")


# ==========================================
# USAGE
# ==========================================

if __name__ == "__main__":
    # Validate config
    ANFISConfig.validate_config()
    
    # Print config
    ANFISConfig.print_config()
    
    # Test helper methods
    test_outputs = [0.2, 0.5, 0.8]
    print("Risk Level Tests:")
    for output in test_outputs:
        risk = ANFISConfig.get_risk_label(output)
        print(f"  Output: {output:.2f} → {risk}")