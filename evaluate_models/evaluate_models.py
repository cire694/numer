
from evaluate import validate
from utils import load_model
from config import Config
from data import get_features
from pathlib import Path
import json
import os



if __name__ == "__main__":

    last_train_era =   # import from .json usually

    model_name = "ensemble_lgbm_20260624_015211"
    
    with open(f"models/{model_name}_config.json", "r") as f:
        config_json = json.load(f)

    config = Config(
        model_name=config_json["model_name"],
        data_version=config_json["data_version"],
        feature_set=config_json["feature_set"],
        model_params=config_json["model_params"],
    )

    features = get_features(config, "train")
    model = load_model(f"models/{model_name}.pkl")

    results = validate(model, features, last_train_era=last_train_era)

    output = {
        "model": model_name,
        "last_train_era": last_train_era,
        **results,
    }
    out_path = Path(f"evaluate_models/{model_name}_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(results)
    print(f"Saved to {out_path}")