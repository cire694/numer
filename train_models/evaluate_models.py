
from evaluate import validate
from utils import load_model
from config import Config
from data import get_features
import json


if __name__ == "__main__":

    last_train_era = 574  # import from .json usually

    with open("models/ensemble_lgbm_20260624_015211_config.json", "r") as f:
        config_json = json.load(f)

    config = Config(
        model_name=config_json["model_name"],
        data_version=config_json["data_version"],
        feature_set=config_json["feature_set"],
        model_params=config_json["model_params"],
    )

    features = get_features(config, "train")
    model = load_model("models/ensemble_lgbm_20260624_015211.pkl")

    results = validate(model, features, last_train_era=last_train_era)
    print(results)