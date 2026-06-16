from typing import Dict, Any
from sklearn.base import BaseEstimator
from data import load_dataset
from models import build_model
from evaluate import era_correlations, summarize
from config import Config


def run(config: Config) -> Dict[str, Any]: 
    """Run training pipeline and return evaluation results."""
    train, features = load_dataset(config, "train")
    val, _ = load_dataset(config, "validation")

    model = build_model(config)
    model.fit(train[features], train["target"])

    preds = model.predict(val[features])
    corrs = era_correlations(preds, val["target"], val["era"])
    return summarize(corrs)