"""
diagnose_ensemble.py — pick apart a trained DynamicEnsemble.

Run this AFTER training (or against a mid-training checkpoint — even a
checkpoint_base_models.pkl with an untrained meta head still lets you
inspect per-tree correlation, just not gate weights yet).

For each era in the given dataframe (typically your validation set), computes:
  1. Each base tree's own correlation with the actual target (is it
     learning anything at all, independent of the gate?)
  2. The gate's mean softmax weight assigned to each tree
  3. The final ensemble's correlation

Produces two plots:
  - era vs per-tree correlation (are some trees consistently dead/noisy?)
  - era vs mean gate weight per tree (is the gate actually discriminating,
    or collapsing to near-uniform / near-one-hot?)

Usage:
    python diagnose_ensemble.py --model-path models/dynamic_ensemble_XXXX.pkl \
        --data-split validation --out-dir diagnostics/
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_model
from data import load_dataset
from config import Config


def per_era_diagnostics(ensemble, df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Compute per-era tree correlations and gate weights.

    Returns a long-format DataFrame: era, model_name, corr, mean_weight.
    """
    ensemble.eval()
    rows = []

    with torch.no_grad():
        # Gate weights for every row at once (cheap — it's just the small head/mlp)
        weights = ensemble._compute_weights(df).cpu().numpy()  # (n_rows, n_models)

        base_preds = []
        for (name, model), features in zip(ensemble.ensemble_models, ensemble.features):
            preds = np.asarray(model.predict(df[features].to_numpy()), dtype=np.float32)
            base_preds.append(preds)
        base_preds = np.stack(base_preds, axis=1)  # (n_rows, n_models)

    model_names = [name for name, _ in ensemble.ensemble_models]
    df = df.copy()
    df["_row_idx"] = np.arange(len(df))

    for era, era_df in df.groupby("era"):
        idx = era_df["_row_idx"].to_numpy()
        y = era_df[target_col].to_numpy()
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            continue  # skip eras with too few valid targets to compute a meaningful corr

        for m, name in enumerate(model_names):
            preds_m = base_preds[idx, m][valid]
            y_valid = y[valid]
            if np.std(preds_m) < 1e-8:
                corr = np.nan  # constant predictions (e.g. a 1-tree stump) -> undefined corr
            else:
                corr = np.corrcoef(preds_m, y_valid)[0, 1]
            mean_weight = weights[idx, m].mean()
            rows.append({
                "era": era, "model": name, "corr": corr, "mean_weight": mean_weight,
            })

    return pd.DataFrame(rows)


def ensemble_per_era_corr(ensemble, df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Overall ensemble correlation per era, for comparison against individual trees."""
    preds = ensemble.predict(df)
    df = df.copy()
    df["_pred"] = preds
    rows = []
    for era, era_df in df.groupby("era"):
        y = era_df[target_col].to_numpy()
        p = era_df["_pred"].to_numpy()
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            continue
        corr = np.corrcoef(p[valid], y[valid])[0, 1]
        rows.append({"era": era, "corr": corr})
    return pd.DataFrame(rows)


def plot_diagnostics(diag_df: pd.DataFrame, ensemble_corr_df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_names = sorted(diag_df["model"].unique())

    # Plot 1: per-tree correlation over eras
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in model_names:
        sub = diag_df[diag_df["model"] == name].sort_values("era")
        ax.plot(sub["era"], sub["corr"], label=name, alpha=0.6, linewidth=1)
    ens = ensemble_corr_df.sort_values("era")
    ax.plot(ens["era"], ens["corr"], label="ENSEMBLE", color="black", linewidth=2.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("era")
    ax.set_ylabel("correlation with target")
    ax.set_title("Per-tree correlation by era (flat-near-zero or NaN = dead tree)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_tree_corr_by_era.png"), dpi=150)
    plt.close(fig)

    # Plot 2: mean gate weight over eras (stacked area — should sum to 1 per era)
    pivot = diag_df.pivot(index="era", columns="model", values="mean_weight").sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(pivot.index, pivot.T.values, labels=pivot.columns, alpha=0.85)
    ax.set_xlabel("era")
    ax.set_ylabel("mean gate weight (softmax, sums to 1)")
    ax.set_title("Gate weight allocation by era (flat bands = gate isn't discriminating)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate_weights_by_era.png"), dpi=150)
    plt.close(fig)

    # Summary table: overall (non-era-specific) health check per tree
    summary = diag_df.groupby("model").agg(
        mean_corr=("corr", "mean"),
        pct_eras_dead=("corr", lambda x: x.isna().mean()),  # NaN corr = constant predictions that era
        mean_weight=("mean_weight", "mean"),
    ).sort_values("mean_weight", ascending=False)
    summary.to_csv(os.path.join(out_dir, "tree_health_summary.csv"))
    print("\n=== Tree health summary ===")
    print(summary.to_string())
    print(f"\nSaved plots + summary to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-split", default="validation", choices=["train", "validation"])
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--out-dir", default="diagnostics")
    args = parser.parse_args()

    ensemble = load_model(args.model_path)

    config_path = args.model_path.replace(".pkl", "_config.json")
    import json
    with open(config_path) as f:
        cfg = json.load(f)
    config = Config(**{k: cfg[k] for k in ("model_name", "data_version", "feature_set", "model_params")})

    df, _ = load_dataset(config, args.data_split)

    print("Computing per-tree, per-era diagnostics (this predicts with all 16 boosters, may take a few minutes)...")
    diag_df = per_era_diagnostics(ensemble, df, target_col=args.target_col)
    ens_corr_df = ensemble_per_era_corr(ensemble, df, target_col=args.target_col)
    plot_diagnostics(diag_df, ens_corr_df, args.out_dir)