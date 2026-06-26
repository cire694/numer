"""Utilities for model persistence and common operations."""
import os
import cloudpickle
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from sklearn.base import BaseEstimator
from config import Config
from typing import Protocol
import numpy as np


class Regressor(Protocol):
    def fit(self, X, y) -> None: ...
    def predict(self, X) -> np.ndarray: ...

def save_model(model: Regressor, config: Config, model_name: Optional[str] = None, last_train_era: Optional[int] = None, output_dir: str = "./models") -> str:
    """Save a trained model and its config metadata to disk.

    Args:
        model: Trained estimator to save.
        config: Configuration object used for training.
        model_name: Optional override for the saved model name.
        last_train_era: Optional last training era for embargo calculation.
        output_dir: Directory where model files are written.

    Returns:
        The file path of the saved model pickle.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    name = model_name or config.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"{name}_{timestamp}.pkl")
    
    with open(model_path, "wb") as f:
        cloudpickle.dump(model, f)
    
    # Save config alongside model
    config_path = os.path.join(output_dir, f"{name}_{timestamp}_config.json")
    config_dict = {
        "model_name": config.model_name,
        "data_version": config.data_version,
        "feature_set": config.feature_set,
        "model_params": config.model_params,
    }
    if last_train_era is not None:
        config_dict["last_train_era"] = last_train_era
    
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Model saved: {model_path}")
    return model_path


def load_model(model_path: str) -> BaseEstimator:
    """Load a trained model from disk.

    Args:
        model_path: Path to the saved pickle file.

    Returns:
        The deserialized estimator.
    """
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
    print(f"Model loaded: {model_path}")
    return model



def get_latest_model(model_name: str, models_dir: str = "./models") -> str:
    """Get the most recently saved model file for a given model name.

    Args:
        model_name: Name of the model to search for.
        models_dir: Directory containing saved model files.

    Returns:
        Path to the most recently saved model file.

    Raises:
        FileNotFoundError: If no matching model files are found.
    """
    import glob
    
    pattern = os.path.join(models_dir, f"{model_name}_*.pkl")
    models = glob.glob(pattern)
    
    if not models:
        raise FileNotFoundError(f"No models found for {model_name}")
    
    # Get the most recent one (by name, since we use timestamp)
    latest = max(models, key=os.path.getctime)
    return latest
