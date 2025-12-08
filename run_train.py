import os
from run import ExperimentConfig
from src.ml.train import train_model

config = ExperimentConfig("my_first_experiment")
config.create_dirs()

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

print("Model saved:", os.path.exists(config.model_path))