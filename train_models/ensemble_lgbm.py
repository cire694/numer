import lightgbm as lgb
from joblib import Parallel, delayed

from data import load_dataset
from config import Config
from train_models.Ensemble import EnsembleModel
from utils import save_model, load_model


if __name__ == "__main__":
    
    config = Config(model_name='none')
    train, features = load_dataset(config, split="train")
    val, _ = load_dataset(config, split="validation")

    params = {
        "num_threads": 4, #cores per model
        "num_leaves": 31,
        "colsample_bytree": 0.1,
        "learning_rate": 0.01,
        "n_estimators": 2000,
        "min_child_samples": 20,
        "verbose": -1,  # suppress LightGBM's per-iteration output
    }

    target_cols = [c for c in train.columns if c.startswith("target")]

    ensemble = EnsembleModel()
    for target_col in target_cols: 
        ensemble.add_model(target_col, lgb.LGBMRegressor(**params))    

    print("running target_ensemble")
    ensemble.fit(train, features, targets=target_cols)

    print("saving model")

    save_model(ensemble, config)
    print("finished saving successfully")

    # loaded = load_model("path.pkl")
    # preds = loaded.predict(live_features)


    
