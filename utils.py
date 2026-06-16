"""Utilities for model persistence and common operations."""
import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from sklearn.base import BaseEstimator
from config import Config


def save_model(model: BaseEstimator, config: Config, model_name: Optional[str] = None, output_dir: str = "./models") -> str:
    """Save trained model to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    name = model_name or config.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"{name}_{timestamp}.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save config alongside model
    config_path = os.path.join(output_dir, f"{name}_{timestamp}_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": config.model_name,
            "data_version": config.data_version,
            "feature_set": config.feature_set,
            "model_params": config.model_params,
        }, f, indent=2)
    
    print(f"Model saved: {model_path}")
    return model_path


def load_model(model_path: str) -> BaseEstimator:
    """Load trained model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded: {model_path}")
    return model


def get_latest_model(model_name: str, models_dir: str = "./models") -> str:
    """Get the most recently saved model of a given type."""
    import glob
    
    pattern = os.path.join(models_dir, f"{model_name}_*.pkl")
    models = glob.glob(pattern)
    
    if not models:
        raise FileNotFoundError(f"No models found for {model_name}")
    
    # Get the most recent one (by name, since we use timestamp)
    latest = max(models, key=os.path.getctime)
    return latest
