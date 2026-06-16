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
    """Train a model and save it."""
    print(f"Training {config.model_name}...")
    results = run(config)
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


def submit(config: Config, model: Optional[BaseEstimator] = None, model_path: Optional[str] = None) -> None:
    """Generate and submit predictions using a trained model."""
    if model is None:
        if model_path is None:
            model_path = get_latest_model(config.model_name)
        model = load_model(model_path)
    
    full_submission_pipeline(model, config)


def main():
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