from typing import Dict
import numpy as np
import pandas as pd


def era_correlations(predictions: np.ndarray, targets: np.ndarray, eras: np.ndarray) -> pd.Series:
    """Compute era-level Spearman rank correlations for a prediction set.

    Args:
        predictions: Predicted values for each row.
        targets: Actual target values for each row.
        eras: Era labels for each row.

    Returns:
        Series of correlation values indexed by era.
    """
    df = pd.DataFrame({"pred": predictions, "target": targets, "era": eras})

    # Rank within each era (vectorized across all eras at once)
    df["pred_rank"] = df.groupby("era")["pred"].rank(pct=True)
    df["target_rank"] = df.groupby("era")["target"].rank(pct=True)

    # Precompute the cross/square terms needed for the correlation formula
    df["xy"] = df["pred_rank"] * df["target_rank"]
    df["x2"] = df["pred_rank"] ** 2
    df["y2"] = df["target_rank"] ** 2

    agg = df.groupby("era").agg(
        n=("pred_rank", "size"),
        sum_x=("pred_rank", "sum"),
        sum_y=("target_rank", "sum"),
        sum_xy=("xy", "sum"),
        sum_x2=("x2", "sum"),
        sum_y2=("y2", "sum"),
    )

    numerator = agg["n"] * agg["sum_xy"] - agg["sum_x"] * agg["sum_y"]
    denominator = np.sqrt(
        (agg["n"] * agg["sum_x2"] - agg["sum_x"] ** 2)
        * (agg["n"] * agg["sum_y2"] - agg["sum_y"] ** 2)
    )

    return numerator / denominator


def summarize(era_corrs: pd.Series) -> Dict[str, float]:
    """Summarize era correlations into key metrics."""
    return {
        "mean_corr": float(era_corrs.mean()),
        "std_corr": float(era_corrs.std()),
        "sharpe": float(era_corrs.mean() / era_corrs.std()),
    }


def validate(model, features: list, last_train_era: int, val_data_path: str = "./data/validation/validation.parquet", downsample=4) -> Dict[str, float]:
    """
    Evaluates a trained model on the validatino set
    
    Args: 
        model: trained model with a .predict() method
        features: list of feature column names
        last_train_era: last era used in training (as int) for embargo
        val_data_path: path to validation parquet file
        downsample: keep every nth era (1 = all eras)

    Returns: 
        Dict with mean_coor, std_coor, sharpe
    """
    validation = pd.read_parquet(
        val_data_path, 
        columns=["era", "target"] + features
    )
    
    validation = validation[validation['era'].isin(validation['era'].unique()[::downsample])]

    # the target is computed over 20 days of future returns
    # we need to skip the first for eras after last training era
    # e.g. train:       ...   0498, 0499, 0500
    #      Embargo:     0501, 0502, 0503, 0504
    #      Validation:  0505, 0506, ...
    last_train_era = last_train_era
    eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]

    validation = validation[~validation["era"].isin(eras_to_embargo)]

    validation["prediction"] = model.predict(validation[features])

    era_corrs = era_correlations(
        validation["prediction"].values, 
        validation["target"].values, 
        validation['era'].values,
    )

    return summarize(era_corrs)


    
