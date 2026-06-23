import lightgbm as lgb

from typing import Dict
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, LinearRegression
from config import Config

MODEL_REGISTRY: Dict[str, type] = {
	"xgboost": XGBRegressor,
	"lgbm": lgb.LGBMRegressor,
	"ridge": Ridge,
	"linear": LinearRegression,
}


def build_model(config: Config) -> BaseEstimator:
	"""Build and return a model instance from the registry.

	Args:
		config: Configuration object with model selection and hyperparameters.

	Returns:
		An initialized scikit-learn estimator.

	Raises:
		ValueError: If the configured model name is not registered or is 'none'.
	"""
	if config.model_name == "none":
		raise ValueError("Cannot build model with model_name='none'. Use this only for submit() with pre-trained models.")
	if config.model_name not in MODEL_REGISTRY:
		raise ValueError(f"Model {config.model_name} not in registry")

	model_class = MODEL_REGISTRY[config.model_name]
	return model_class(**config.model_params)

