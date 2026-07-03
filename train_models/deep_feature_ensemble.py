import lightgbm as lgb
import json
from pathlib import Path
from data import load_dataset, load_feature_groups
from config import Config
from train_models.Ensemble import EnsembleModel
from utils import save_model
from evaluate import validate

"""
Difference: 
    - more regularization, larger trees
    - Each tree is trained on different features, not different targets
    - training on all rows
"""
 
if __name__ == "__main__":
    
    config = Config(model_name='deep_feature_ensemble', feature_set="all")
    train, features = load_dataset(config, split="train")
    print("finish loading train")
    val, _ = load_dataset(config, split="validation") #needed for early stopping

    print("finish loading val")
    params = {
        "num_threads": 4,
        "num_leaves": 64,
        "colsample_bytree": 0.1,
        "subsample": 0.7, # fraction of all rows each tree sees 
        "subsample_freq": 1, # apply bagging every iteration
        "learning_rate": 0.001,
        "n_estimators": 10_000,
        "min_child_samples": 1000,
        "verbose": -1,
        "reg_lambda": 5.0, 
    }

    feature_sets = load_feature_groups()

    print("finish loading feature groups")
    feature_group_names = [
        'intelligence', 'charisma', 'strength', 'dexterity', 'constitution',
        'wisdom', 'agility', 'serenity', 'sunshine', 'rain', 'midnight', 'faith'
    ]

    missing = [g for g in feature_group_names if g not in feature_sets]
    if missing:
        raise KeyError(f"These groups aren't in feature_sets for this data version: {missing}")
    
    loaded_features = set(features)  # columns actually present in `train`/`val`
    per_model_features = []
    for g in feature_group_names:
        group_feats = [f for f in feature_sets[g] if f in loaded_features]
        if not group_feats:
            raise ValueError(f"Group '{g}' has zero overlap with loaded features — check feature_size used in load_dataset")
        per_model_features.append(group_feats)

    ensemble = EnsembleModel()
    for groups in feature_group_names: 
        ensemble.add_model(groups, lgb.LGBMRegressor(**params))    

    print("finish smoke test")
    print("running feature_ensemble: each tree trains on a different feature group")
    print(f"Number of features: {len(feature_group_names)}")
    early_stop_callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
    ]
    ensemble.fit(
        train,
        per_model_features, 
        targets="target", #train on default target
        val=val,
        callbacks=early_stop_callbacks,
        n_jobs=4
    )

    print("saving model")
    
    last_train_era = int(train["era"].unique()[-1])
    model_path = save_model(ensemble, config, last_train_era=last_train_era)
    print("finished saving successfully")

    # ── Evaluate the model ───────────────────────────────────────────────────
    # Use the trained ensemble directly (already in memory)
    model_name_with_timestamp = Path(model_path).stem
    
    results = validate(ensemble, features, last_train_era=last_train_era, downsample=1)
    
    # Save results to JSON
    output = {
        "model": model_name_with_timestamp,
        "last_train_era": last_train_era,
        **results,
    }
    out_path = Path(f"evaluate_models/{model_name_with_timestamp}_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(results)
    print(f"Saved to {out_path}")



    
