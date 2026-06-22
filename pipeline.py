from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator
from data import load_dataset
from models import build_model
from evaluate import validate
from config import Config


def run(config: Config) -> Tuple[Dict[str, Any], BaseEstimator]: 
    """Run the training pipeline and evaluate validation performance.

    Args:
        config: Configuration object with model and data parameters.

    Returns:
        A dictionary with mean_corr, std_corr, sharpe, on the validation set
        The trained model

    """
    train, features = load_dataset(config, "train")

    model = build_model(config)
    model.fit(train[features], train["target"])
    
    last_train_era = int(train["era"].unique()[-1])
    return validate(model, features, last_train_era, downsample=config.val_downsample), model