from typing import Tuple, Optional
import argparse
import pandas as pd
from sklearn.base import BaseEstimator
from config import Config
from pipeline import run
from submission import full_submission_pipeline
from models import build_model
from utils import save_model, load_model, get_latest_model
from data import load_dataset


def train(config: Config) -> Tuple[BaseEstimator, str]:
    """Train a model, evaluate validation performance, and save the final model.

    Args:
        config: Configuration object with model and data settings.

    Returns:
        A tuple of (trained model, saved model path).
    
    Raises:
        ValueError: If model_name is 'none'.
    """
    if config.model_name == "none":
        raise ValueError("Cannot train with model_name='none'. model_name='none' is for submit() only.")
    
    print(f"Training {config.model_name}...")
    results, trained_model = run(config)
    print(f"Results: {results}")
    
    # Train final model on all data for submission
    print("Training final model on all available data...")
    train_data, features = load_dataset(config, "train")
    val_data, _ = load_dataset(config, "validation")
    
    # Combine train + validation for final model
    final_train = pd.concat([train_data, val_data], ignore_index=True)
    
    model = build_model(config)
    model.fit(final_train[features], final_train["target"])
    
    model_path = save_model(model, config)
    return model, model_path


def submit(config: Optional[Config] = None, model: Optional[BaseEstimator] = None, model_path: Optional[str] = None) -> None:
    """Generate and submit predictions using a trained model or a saved model file.

    Args:
        config: Optional Configuration object with submission settings. If not provided,
                will be loaded from the model's saved config.json file.
        model: Optional trained estimator to use directly.
        model_path: Optional path to a saved model pickle.
    """
    import json
    import os
    
    if model is None:
        if model_path is None:
            if config is None:
                raise ValueError("Must provide either config, model, or model_path")
            model_path = get_latest_model(config.model_name)
        model = load_model(model_path)
        
        # If config not provided, load it from the model's config.json
        if config is None:
            config_path = model_path.replace(".pkl", "_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    config = Config(
                        model_name=config_dict["model_name"],
                        data_version=config_dict["data_version"],
                        feature_set=config_dict["feature_set"],
                        model_params=config_dict["model_params"]
                    )
            else:
                raise ValueError(f"Config file not found at {config_path}")
    
    if config is None:
        raise ValueError("Could not determine config. Provide config or model_path with config.json")
    
    full_submission_pipeline(model, config)


def main():
    """Parse command-line arguments and execute the Numerai pipeline."""
    
    parser = argparse.ArgumentParser(description="Numerai pipeline")
    parser.add_argument("--mode", choices=["train", "submit", "train-submit"], 
                        default="train-submit", help="Pipeline mode")
    parser.add_argument("--model", default="ridge", help="Model name")
    parser.add_argument("--data-version", default="v5.2", help="Data version")
    parser.add_argument("--feature-set", default="small", help="Feature set")
    
    args = parser.parse_args()
    
    config = Config(
        model_name=args.model,
        data_version=args.data_version,
        feature_set=args.feature_set
    )
    
    if args.mode in ["train", "train-submit"]:
        model, model_path = train(config)
    
    if args.mode in ["submit", "train-submit"]:
        if args.mode == "submit":
            submit(config)
        else:
            submit(config, model=model)


if __name__ == "__main__":
    main()