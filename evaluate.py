from typing import Dict
import numpy as np
import pandas as pd


def era_correlations(predictions: np.ndarray, targets: np.ndarray, eras: np.ndarray) -> pd.Series:
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