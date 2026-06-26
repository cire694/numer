from typing import Optional
from sklearn.base import BaseEstimator
from config import Config
from submission import full_submission_pipeline
from utils import load_model, get_latest_model



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

if __name__ == "__main__":
    
    #submit whatever
    print("nothing to submit")