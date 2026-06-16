"""Submission module for Numerai predictions."""
from typing import Optional, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from numerapi import NumerAPI
from data import load_dataset
from models import build_model
from config import Config


def generate_predictions(model: BaseEstimator, config: Config, data_split: str = "live") -> Tuple[pd.DataFrame, pd.Series]:
    """Generate predictions for a given split (typically 'live')."""
    live_data, features = load_dataset(config, data_split)
    predictions = model.predict(live_data[features])
    return live_data[["era"]].copy(), predictions


def format_submission(era_df: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Format predictions into Numerai submission format."""
    submission = era_df.copy()
    submission["prediction"] = predictions
    return submission


def submit_predictions(submission_df: pd.DataFrame, config: Config, model_name: Optional[str] = None) -> str:
    """Submit predictions to Numerai API."""
    napi = NumerAPI()
    
    # Convert to CSV format (required by API)
    csv_data = submission_df.to_csv(index=False)
    
    # Submit using version and model name
    version = config.data_version
    name = model_name or config.model_name
    
    submission_id = napi.upload_predictions(csv_data, model_name=name, version=version)
    return submission_id


def full_submission_pipeline(model: BaseEstimator, config: Config, model_name: Optional[str] = None) -> str:
    """Execute full submission pipeline."""
    print(f"Generating predictions for {config.data_version}...")
    era_df, predictions = generate_predictions(model, config)
    
    print("Formatting submission...")
    submission_df = format_submission(era_df, predictions)
    
    print("Submitting to Numerai...")
    submission_id = submit_predictions(submission_df, config, model_name)
    
    print(f"Submission successful! ID: {submission_id}")
    return submission_id
