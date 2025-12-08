import os
import json
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ml.train import train_model
from src.ml.predict import predict_on_data
from src.ml.eval import evaluate_on_data

def save_experiment_log(config, quick_eval_metrics, eval_metrics):
    log = {
        "experiment_name": config.experiment_name,
        "timestamp": config.timestamp,
        "paths": {
            "raw_data": config.raw_data_path,
            "features_csv": config.features_path,
            "model": config.model_path,
            "predictions": config.predictions_path,
            "evaluation_dir": config.eval_dir
        },
        "feature_engineering": {
            "target_type": config.target_type,
            "target_window": config.target_window,
            "rolling_window": config.rolling_window,
            "n_features": config.n_features,
            "n_samples": config.n_samples,
            "class_balance": config.class_balance
        },
        "model": config.model_params,
        "quick_eval": quick_eval_metrics,
        "evaluation": eval_metrics
    }

    log_dir = os.path.join(config.eval_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"experiment_log_{config.timestamp}.json")

    with open(log_path, "w") as f:
        json.dump(log, f, indent=4)

    print(f"[INFO] Experiment details saved to {log_path}")

# EXPERIMENT CONFIGURATION

class ExperimentConfig:
    """
    Central configuration for experiment.
    All paths and hyperparameters defined here.
    Optimized for stable predictive maintenance performance.
    """
    def __init__(self, experiment_name, base_dir="C:/Users/Korisnik/py/manufacturing"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        self.raw_data_path = os.path.join(base_dir, "data/raw/logs_all_machines.csv")
        self.experiment_dir = os.path.join(base_dir, "experiments", f"{experiment_name}_{self.timestamp}")
        self.model_path = os.path.join(self.experiment_dir, "model.pkl")
        self.features_path = os.path.join(self.experiment_dir, "features.csv")
        self.predictions_path = os.path.join(self.experiment_dir, "predictions.csv")
        self.eval_dir = os.path.join(self.experiment_dir, "evaluation")
        
        # === FEATURE ENGINEERING CONFIG ===
        self.target_type = 'windowed'
        self.target_window = 7          # recommended 7-9 cycles ahead
        self.rolling_window = 30      # recommended 19-24 cycles for stability
        self.rolling_cols = ['cycle_time', 'temperature', 'vibration', 'pressure']
        self.lag_cols = ['cycle_time', 'temperature', 'vibration', 'pressure']
        self.lag_periods = [1, 3, 5]
        self.cat_cols = ['operator', 'maintenance_type']
        
        # === MODEL HYPERPARAMETERS ===
        self.model_params = {
            'n_estimators': 115,        # slightly increased for stability
            'max_depth': 9,
            'learning_rate': 0.12,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'scale_pos_weight': 13,     # balance rare downtime events
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # === TRAINING CONFIG ===
        self.test_size = 0.2
        self.oversample = True         # controlled oversampling
        self.optimize_threshold = True # plan to tune threshold based on Precision-Recall trade-off
        
    def create_dirs(self):
        """Create experiment directories."""
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        
    def print_config(self):
        """Print experiment configuration."""
        print("\n" + "="*70)
        print(f"EXPERIMENT: {self.experiment_name}")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Experiment dir: {self.experiment_dir}")
        print(f"\n--- DATA ---")
        print(f"Raw data: {self.raw_data_path}")
        print(f"\n--- FEATURE ENGINEERING ---")
        print(f"Target type: {self.target_type}")
        if self.target_type == 'windowed':
            print(f"Target window: {self.target_window}")
        print(f"Rolling window: {self.rolling_window}")
        print(f"\n--- MODEL ---")
        print(f"N estimators: {self.model_params['n_estimators']}")
        print(f"Max depth: {self.model_params['max_depth']}")
        print(f"Learning rate: {self.model_params['learning_rate']}")
        print(f"Scale pos weight: {self.model_params['scale_pos_weight']}")
        print(f"Oversample: {self.oversample}")
        print(f"Optimize threshold: {self.optimize_threshold}")
        print("="*70 + "\n")

# RUN EXPERIMENT

def run_experiment(config):
    """
    Run complete experiment: train → predict → evaluate.
    
    Args:
        config: ExperimentConfig object
    """
    config.create_dirs()
    config.print_config()
    
    # === STEP 1: TRAIN ===
    print("\n" + "🚂 "*20)
    print("STEP 1: TRAINING")
    print("🚂 "*20 + "\n")
    
    clf, X_test, y_test = train_model(
        data_path=config.raw_data_path,
        output_model_path=config.model_path,
        output_data_path=config.features_path,
        target_type=config.target_type,
        target_window=config.target_window,
        rolling_window=config.rolling_window,
        test_size=config.test_size,
        oversample=config.oversample,
        model_params=config.model_params
    )
    
    # === STEP 2: PREDICT ===
    print("\n" + "🔮 "*20)
    print("STEP 2: PREDICTION")
    print("🔮 "*20 + "\n")
    
    df_predictions = predict_on_data(
        model_path=config.model_path,
        data_path=config.raw_data_path,
        output_path=config.predictions_path,
        target_type=config.target_type,
        target_window=config.target_window,
        rolling_window=config.rolling_window
    )
    
    # === STEP 3: EVALUATE ===
    print("\n" + "📊 "*20)
    print("STEP 3: EVALUATION")
    print("📊 "*20 + "\n")
    
    evaluate_on_data(
        model_path=config.model_path,
        test_csv=config.raw_data_path,
        output_dir=config.eval_dir,
        target_type=config.target_type,
        target_window=config.target_window,
        rolling_window=config.rolling_window
    )
    
    # === SUMMARY ===
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"Results saved to: {config.experiment_dir}")
    print(f"  - Model: {config.model_path}")
    print(f"  - Features: {config.features_path}")
    print(f"  - Predictions: {config.predictions_path}")
    print(f"  - Evaluation: {config.eval_dir}")
    print("="*70 + "\n")

# GRID SEARCH

def run_grid_search(base_config, param_grid):
    """
    Run grid search over hyperparameters.
    
    Args:
        base_config: ExperimentConfig object (base configuration)
        param_grid: Dict of parameters to search
                   Example: {'target_window': [3, 5, 10], 
                            'rolling_window': [10, 20, 30]}
    
    Returns:
        List of (config, results) tuples
    """
    import itertools
    import json
    
    print("\n" + "🔍 "*20)
    print("GRID SEARCH MODE")
    print("🔍 "*20)
    print(f"\nParameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"\nTotal experiments: {len(combinations)}\n")
    
    all_results = []
    
    for i, combo in enumerate(combinations, 1):
        print("\n" + "="*70)
        print(f"EXPERIMENT {i}/{len(combinations)}")
        print("="*70)
        
        # Create config for this combination
        config = ExperimentConfig(
            experiment_name=f"{base_config.experiment_name}_grid_{i}",
            base_dir=base_config.base_dir
        )
        
        # Update parameters from grid
        for key, value in zip(keys, combo):
            if hasattr(config, key):
                setattr(config, key, value)
            elif key in ['n_estimators', 'max_depth', 'learning_rate']:
                config.model_params[key] = value
        
        # Run experiment
        try:
            run_experiment(config)
            
            # Load results
            metrics_path = os.path.join(config.eval_dir, "metrics.json")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            result = {
                'config': {k: v for k, v in zip(keys, combo)},
                'metrics': metrics,
                'experiment_dir': config.experiment_dir
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Experiment {i} failed: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE")
    print("="*70)
    print(f"\nSuccessful experiments: {len(all_results)}/{len(combinations)}\n")
    
    # Sort by ROC-AUC
    all_results.sort(key=lambda x: x['metrics']['roc_auc'], reverse=True)
    
    print("TOP 5 CONFIGURATIONS (by ROC-AUC):")
    print("-"*70)
    for i, result in enumerate(all_results[:5], 1):
        print(f"\n{i}. ROC-AUC: {result['metrics']['roc_auc']:.4f}, F1: {result['metrics']['f1']:.4f}")
        print(f"   Config: {result['config']}")
        print(f"   Dir: {result['experiment_dir']}")
    print("="*70 + "\n")
    
    return all_results


# =============================================================================
# RANDOM SEARCH
# =============================================================================

def run_random_search(base_config, param_distributions, n_iter=10):
    """
    Run random search over hyperparameters.
    
    Args:
        base_config: ExperimentConfig object
        param_distributions: Dict of parameters with ranges
                            Example: {'n_estimators': [100, 300],
                                     'max_depth': [4, 8],
                                     'learning_rate': [0.01, 0.1]}
        n_iter: Number of random configurations to try
    
    Returns:
        List of (config, results) tuples
    """
    import random
    import json
    
    print("\n" + "🎲 "*20)
    print("RANDOM SEARCH MODE")
    print("🎲 "*20)
    print(f"\nParameter distributions:")
    for key, (low, high) in param_distributions.items():
        print(f"  {key}: [{low}, {high}]")
    print(f"\nIterations: {n_iter}\n")
    
    all_results = []
    
    for i in range(1, n_iter + 1):
        print("\n" + "="*70)
        print(f"EXPERIMENT {i}/{n_iter}")
        print("="*70)
        
        # Create config
        config = ExperimentConfig(
            experiment_name=f"{base_config.experiment_name}_random_{i}",
            base_dir=base_config.base_dir
        )
        
        # Sample random parameters
        sampled_params = {}
        for key, (low, high) in param_distributions.items():
            if isinstance(low, int) and isinstance(high, int):
                value = random.randint(low, high)
            else:
                value = random.uniform(low, high)
            sampled_params[key] = value
            
            # Update config
            if hasattr(config, key):
                setattr(config, key, value)
            elif key in config.model_params:
                config.model_params[key] = value
        
        print(f"Sampled params: {sampled_params}\n")
        
        # Run experiment
        try:
            run_experiment(config)
            
            # Load results
            metrics_path = os.path.join(config.eval_dir, "metrics.json")
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            result = {
                'config': sampled_params,
                'metrics': metrics,
                'experiment_dir': config.experiment_dir
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Experiment {i} failed: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("RANDOM SEARCH COMPLETE")
    print("="*70)
    print(f"\nSuccessful experiments: {len(all_results)}/{n_iter}\n")
    
    # Sort by ROC-AUC
    all_results.sort(key=lambda x: x['metrics']['roc_auc'], reverse=True)
    
    print("TOP 5 CONFIGURATIONS (by ROC-AUC):")
    print("-"*70)
    for i, result in enumerate(all_results[:5], 1):
        print(f"\n{i}. ROC-AUC: {result['metrics']['roc_auc']:.4f}, F1: {result['metrics']['f1']:.4f}")
        print(f"   Config: {result['config']}")
        print(f"   Dir: {result['experiment_dir']}")
    print("="*70 + "\n")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run ML experiment workflow")
    
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name of experiment (results saved to experiments/{name}_{timestamp}/)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override default data path")
    parser.add_argument("--base_dir", type=str, 
                        default="C:/Users/Korisnik/py/manufacturing",
                        help="Base directory of project")
    
    # Search modes
    parser.add_argument("--grid_search", action='store_true',
                        help="Run grid search")
    parser.add_argument("--random_search", action='store_true',
                        help="Run random search")
    parser.add_argument("--n_iter", type=int, default=10,
                        help="Number of iterations for random search")
    
    args = parser.parse_args()
    
    # Create base config
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        base_dir=args.base_dir
    )
    
    # Override data path if provided
    if args.data_path:
        config.raw_data_path = args.data_path
    
    # === SINGLE EXPERIMENT MODE ===
    if not args.grid_search and not args.random_search:
        run_experiment(config)
    
    # === GRID SEARCH MODE ===
    elif args.grid_search:
        param_grid = {
            'target_window': [3, 5, 10],
            'rolling_window': [10, 20, 30],
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8]
        }
        run_grid_search(config, param_grid)
    
    # === RANDOM SEARCH MODE ===
    elif args.random_search:
        param_distributions = {
            'target_window': [3, 10],           # random int between 3-10
            'rolling_window': [10, 30],         # random int between 10-30
            'n_estimators': [100, 300],         # random int between 100-300
            'max_depth': [4, 10],               # random int between 4-10
            'learning_rate': [0.01, 0.15]       # random float between 0.01-0.15
        }
        run_random_search(config, param_distributions, n_iter=args.n_iter)

if __name__ == "__main__":
    main()