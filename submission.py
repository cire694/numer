"""Submission module for Numerai predictions."""
from typing import Optional, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from numerapi import NumerAPI
from data import load_dataset
from models import build_model
from config import Config


def generate_predictions(model: BaseEstimator, config: Config, data_split: str = "live") -> Tuple[pd.DataFrame, pd.Series]:
    """Generate predictions for a Numerai data split.

    Args:
        model: Trained estimator used to predict.
        config: Configuration object with feature and data settings.
        data_split: Data split to predict on, usually 'live'.

    Returns:
        A tuple of (era dataframe, prediction series).
    """
    live_data, features = load_dataset(config, data_split)
    predictions = model.predict(live_data[features])
    return live_data[["era"]].copy(), predictions


def format_submission(era_df: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Format a prediction series into a Numerai submission DataFrame.

    Args:
        era_df: DataFrame containing era identifiers.
        predictions: Series of predicted target values.

    Returns:
        A submission-ready DataFrame with era and prediction columns.
    """
    submission = era_df.copy()
    submission["prediction"] = predictions
    return submission


def submit_predictions(submission_df: pd.DataFrame, config: Config, model_name: Optional[str] = None) -> str:
    """Submit a formatted submission DataFrame to the Numerai API.

    Args:
        submission_df: DataFrame in Numerai submission format.
        config: Configuration object providing version metadata.
        model_name: Optional submission model name to override config.

    Returns:
        The Numerai submission ID string.
    """
    napi = NumerAPI()
    
    # Convert to CSV format (required by API)
    csv_data = submission_df.to_csv(index=False)
    
    # Submit using version and model name
    version = config.data_version
    name = model_name or config.model_name
    
    submission_id = napi.upload_predictions(csv_data, model_name=name, version=version)
    return submission_id


def full_submission_pipeline(model: BaseEstimator, config: Config, model_name: Optional[str] = None) -> str:
    """Execute the full submission pipeline: generate, format, and submit.

    Args:
        model: Trained estimator used for live predictions.
        config: Configuration object with model and data settings.
        model_name: Optional model name override for the submission.

    Returns:
        The Numerai submission ID string.
    """
    print(f"Generating predictions for {config.data_version}...")
    era_df, predictions = generate_predictions(model, config)
    
    print("Formatting submission...")
    submission_df = format_submission(era_df, predictions)
    
    print("Submitting to Numerai...")
    submission_id = submit_predictions(submission_df, config, model_name)
    
    print(f"Submission successful! ID: {submission_id}")
    return submission_id
