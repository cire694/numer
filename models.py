from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, LinearRegression

MODEL_REGISTRY = {
    "xgboost": XGBRegressor,
    "ridge": Ridge, 
    "linear": LinearRegression,
}

def build_model(config): 
    model_class = MODEL_REGISTRY[config.model_name]
    return model_class(**config.model_params)