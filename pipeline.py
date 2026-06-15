from data import load_dataset
from models import build_model
from evaluate import era_correlations, summarize

def run(config): 
    train, features = load_dataset(config, "train")
    val, _ = load_dataset(config, "validation")

    model = build_model(config)
    model.fit(train[features], train["target"])

    preds = model.predict(val[features])
    corrs = era_correlations(preds, val["target"], val["era"])
    return summarize(corrs)