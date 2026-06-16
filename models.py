from typing import Dict, Type
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, LinearRegression
from config import Config


MODEL_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    "xgboost": XGBRegressor,
    "ridge": Ridge, 
    "linear": LinearRegression,
}


def build_model(config: Config) -> BaseEstimator: 
    """Build and return a model instance from the registry."""
    if config.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {config.model_name} not in registry")
    
    model_class = MODEL_REGISTRY[config.model_name]
    return model_class(**config.model_params)