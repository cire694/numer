import json
import lightgbm as lgb
from data import load_dataset
from config import Config
from evaluate import era_correlations, summarize
from utils import load_model, get_latest_model, load_model_with_config

if __name__ == "__main__":
    config = Config(model_name="none")

    # load data
    train, features = load_dataset(config, split="train")
    val, _ = load_dataset(config, split="validation")

    # load feature group metadata
    import os
    json_path = os.path.join(
        os.path.expanduser(config.data_dir), "train", "features.json"
    )
    feature_metadata = json.load(open(json_path))

    # skip the non-thematic keys
    skip = {"small", "medium", "all", "v2_equivalent_features",
            "v3_equivalent_features", "fncv3_features"}
    feature_groups = {
        k: v for k, v in feature_metadata["feature_sets"].items()
        if k not in skip
    }

    params = {
        "num_threads": 4,
        "num_leaves": 31,
        "colsample_bytree": 0.1,
        "learning_rate": 0.01,
        "n_estimators": 2000,
        "min_child_samples": 20,
        "verbose": -1,
    }

    last_train_era = int(train["era"].unique()[-1])

    # train one model per group and evaluate
    group_results = {}
    for group_name, group_features in feature_groups.items():
        # only use features that actually exist in the loaded dataset
        # (some groups may reference features not in your feature_set)
        available = [f for f in group_features if f in train.columns]
        if not available:
            print(f"{group_name}: no features available in current feature set, skipping")
            continue

        print(f"\n[{group_name}] training on {len(available)} features...")
        model = lgb.LGBMRegressor(**params)
        model.fit(train[available], train["target"])

        preds = model.predict(val[available])
        corrs = era_correlations(preds, val["target"].values, val["era"].values)
        group_results[group_name] = {
            **summarize(corrs),
            "n_features": len(available),
        }
        print(f"[{group_name}] {group_results[group_name]}")

    # summary sorted by sharpe
    print("\n=== Per-group results (sorted by Sharpe) ===")
    for name, res in sorted(group_results.items(),
                            key=lambda x: x[1]["sharpe"], reverse=True):
        print(
            f"{name:20s}  "
            f"sharpe={res['sharpe']:.3f}  "
            f"mean_corr={res['mean_corr']:.4f}  "
            f"std_corr={res['std_corr']:.4f}  "
            f"n_features={res['n_features']}"
        )

    # save results for reference
    with open("./evaluate_models/ensemble_group_results.json", "w") as f:
        json.dump(group_results, f, indent=2)
    print("\nSaved to ./evaluate_models/ensemble_group_results.json")