import argparse
import glob
import gc
import os
import time
import numpy as np
import pandas as pd
import cloudpickle
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Callable, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed

from data import load_dataset, load_feature_groups
from config import Config
from utils import save_model

# NOTE: fit_single is no longer imported from train_models.Ensemble.
# It's been replaced by fit_single_resumable() below, which adds
# mid-training (per-iteration) checkpointing so a single slow model
# doesn't lose all its progress if the SLURM job times out.

"""
[... unchanged module docstring / design notes from the original file ...]
"""


# ──────────────────────────────────────────────────────────────────
# NEW: resumable, mid-training-checkpointed single-model fit
# ──────────────────────────────────────────────────────────────────

def _iteration_checkpoint_callback(name: str, checkpoint_dir: Optional[str], every_n_iters: int = 250):
    """LightGBM training callback that periodically dumps the in-progress
    booster to disk. This is what lets you resume a single very long model
    (e.g. n_estimators=10_000) mid-training instead of only at model
    boundaries.
    """
    def _callback(env):
        if checkpoint_dir is None:
            return
        it = env.iteration + 1  # human-readable, 1-indexed
        is_last = it == env.end_iteration
        if it % every_n_iters == 0 or is_last:
            path = os.path.join(checkpoint_dir, f"partial_{name}.txt")
            env.model.save_model(path, num_iteration=it)
            print(f"[base models] [{name}] iter checkpoint saved: {it}/{env.end_iteration}", flush=True)
    _callback.order = 10
    return _callback


def fit_single_resumable(
    name: str,
    model: lgb.LGBMRegressor,
    train: pd.DataFrame,
    features: List[str],
    target_col: str,
    val: Optional[pd.DataFrame],
    callbacks: Optional[List[Callable]],
    checkpoint_dir: Optional[str],
    checkpoint_every_iters: int = 250,
) -> lgb.LGBMRegressor:
    """Fit one LightGBM base model with mid-training checkpoint/resume.

    On call, checks for an existing `partial_{name}.txt` booster dump in
    checkpoint_dir. If found, resumes boosting from wherever it left off
    (via `init_model`), rather than restarting from iteration 0.
    """
    t0 = time.time()
    total_estimators = model.get_params()["n_estimators"]
    partial_path = os.path.join(checkpoint_dir, f"partial_{name}.txt") if checkpoint_dir else None

    init_model = None
    completed_iters = 0
    if partial_path and os.path.exists(partial_path):
        booster = lgb.Booster(model_file=partial_path)
        completed_iters = booster.current_iteration()
        if completed_iters >= total_estimators:
            print(f"[base models] [{name}] partial checkpoint already complete "
                  f"({completed_iters}/{total_estimators}); reusing without retraining.", flush=True)
            model._Booster = booster
            model.fitted_ = True
            model.set_params(n_estimators=total_estimators)
            return model
        init_model = partial_path
        print(f"[base models] [{name}] RESUMING from partial checkpoint: "
              f"{completed_iters}/{total_estimators} iterations already done "
              f"({completed_iters / total_estimators:.1%}).", flush=True)
    else:
        print(f"[base models] [{name}] starting fresh: 0/{total_estimators} iterations.", flush=True)

    remaining = total_estimators - completed_iters
    model.set_params(n_estimators=remaining)

    cb_list = list(callbacks) if callbacks else []
    if checkpoint_dir is not None:
        cb_list.append(_iteration_checkpoint_callback(name, checkpoint_dir, checkpoint_every_iters))

    X = train[features]
    y = train[target_col]
    fit_kwargs: dict = {}
    if val is not None and target_col in val.columns:
        fit_kwargs["eval_set"] = [(val[features], val[target_col])]
    if init_model is not None:
        fit_kwargs["init_model"] = init_model

    print(f"[base models] [{name}] fitting {remaining} rounds "
          f"on {len(X):,} rows x {len(features)} features...", flush=True)

    model.fit(X, y, callbacks=cb_list, **fit_kwargs)
    model.set_params(n_estimators=total_estimators)  # restore for bookkeeping/reuse

    # Clean up the partial file now that the model is fully trained —
    # avoids a stale checkpoint confusing a later independent run.
    if partial_path and os.path.exists(partial_path):
        try:
            os.remove(partial_path)
        except OSError:
            pass

    elapsed = time.time() - t0
    print(f"[base models] [{name}] DONE: {total_estimators}/{total_estimators} iterations "
          f"in {elapsed/60:.1f} min.", flush=True)
    return model


def carve_era_holdout(df: pd.DataFrame, frac: float, era_col: str = "era"):
    """Split off a small, chronologically-latest slice of eras to use purely
    as an early-stopping signal for the base LightGBM models.

    Splitting by era (not by row) avoids leaking adjacent-era correlation
    between the "training" and "holdout" portions, matching how the
    train/validation split already works elsewhere in this pipeline.

    Args:
        df: Frame to split (e.g. the combined train+validation frame).
        frac: Fraction of *eras* (not rows) to hold out, e.g. 0.03 for 3%.
              Pass 0 (or None) to disable — returns (df, None).
        era_col: Name of the era column.

    Returns:
        (fit_part, holdout_part) — holdout_part is None if frac <= 0.
    """
    if not frac or frac <= 0:
        return df, None
    eras = np.sort(df[era_col].unique())
    n_holdout_eras = max(1, int(round(len(eras) * frac)))
    holdout_eras = set(eras[-n_holdout_eras:])
    holdout_part = df[df[era_col].isin(holdout_eras)]
    fit_part = df[~df[era_col].isin(holdout_eras)]
    print(f"[early stop] holding out {n_holdout_eras}/{len(eras)} eras "
          f"({len(holdout_part):,} rows) purely for early-stopping signal; "
          f"{len(fit_part):,} rows remain for base-model fitting. "
          f"(meta head still trains on all {len(df):,} rows.)", flush=True)
    return fit_part, holdout_part


class DynamicEnsemble(nn.Module):
    """Flexible ensemble of heterogeneous base models.

    Each base model is trained independently, then a neural meta head learns
    instance-specific weights for those base model predictions.
    """

    def __init__(self, embedding_dim: int = 2) -> None:
        super().__init__()
        self.ensemble_models: List[tuple[str, Any]] = []
        self.features: Optional[List[List[str]]] = None
        self.meta_features: Optional[List[str]] = None
        self.embedding_dim = embedding_dim
        self.head: Optional[nn.Embedding] = None
        self.mlp: Optional[nn.Module] = None
        self.embed_drop: Optional[nn.Dropout] = None
        self._meta_head_epochs_trained: int = 0
        self.device: torch.device = torch.device("cpu")
        self.register_buffer("offsets", torch.empty(0, dtype=torch.long))

    def set_device(self, use_cuda: bool = False) -> None:
        """Move the meta head (embeddings + MLP) to GPU, if requested and
        available. Off by default — the meta head is small enough that GPU
        transfer overhead can outweigh any compute savings; this exists so
        it's easy to experiment with on a large enough dataset.
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[meta head] using CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            if use_cuda and not torch.cuda.is_available():
                print("[meta head] --use-cuda was set but no CUDA device is visible; falling back to CPU.", flush=True)
            self.device = torch.device("cpu")
        self.to(self.device)

    def add_model(self, name: str, model: Any) -> "DynamicEnsemble":
        """Register a base model in the ensemble."""
        self.ensemble_models.append((name, model))
        return self

    def _build_meta_head(
        self,
        num_meta_features: int,
        num_models: int,
        hidden_dim: int = 64,
        embed_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ) -> None:
        self.head = nn.Embedding(5 * num_meta_features, self.embedding_dim)
        self.register_buffer("offsets", torch.arange(0, num_meta_features * 5, 5, dtype=torch.long))
        self.embed_drop = nn.Dropout(embed_dropout) if embed_dropout > 0.0 else None
        layers = [nn.Linear(num_meta_features * self.embedding_dim, hidden_dim), nn.ReLU()]
        if hidden_dropout > 0.0:
            layers.append(nn.Dropout(hidden_dropout))
        layers.append(nn.Linear(hidden_dim, num_models))
        self.mlp = nn.Sequential(*layers)

    def _meta_feature_indices(self, X: pd.DataFrame) -> torch.LongTensor:
        if self.meta_features is None:
            raise RuntimeError("DynamicEnsemble meta features are not initialized.")
        x = X.loc[:, self.meta_features].fillna(2).astype(np.int64).to_numpy(copy=True)
        if x.max() >= 5 or x.min() < 0:
            raise ValueError("Meta feature values must be in [0, 4].")
        indices = torch.from_numpy(x).to(self.offsets.device)
        return indices + self.offsets

    def _encode_meta_features(self, X: pd.DataFrame) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("DynamicEnsemble embedding head is not initialized.")
        indices = self._meta_feature_indices(X)
        embeddings = self.head(indices)
        if self.embed_drop is not None:
            embeddings = self.embed_drop(embeddings)
        return embeddings.view(embeddings.shape[0], -1)

    def _compute_weights(self, X: pd.DataFrame) -> torch.Tensor:
        if self.mlp is None:
            raise RuntimeError("DynamicEnsemble meta MLP is not initialized.")
        embeddings = self._encode_meta_features(X)
        logits = self.mlp(embeddings)
        return torch.softmax(logits, dim=1)

    def _train_meta_head(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_X: Optional[pd.DataFrame] = None,
        val_y: Optional[pd.Series] = None,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 4096,
        entropy_coef: float = 0.02,
        weight_decay_head: float = 1e-2,
        weight_decay_mlp: float = 1e-4,
        start_epoch: int = 1,
        optimizer_state: Optional[dict] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1,
    ) -> None:
        if self.mlp is None or self.head is None:
            raise RuntimeError("DynamicEnsemble meta head must be initialized before training.")
        
        if start_epoch > epochs:
            print(f"[meta head] already trained through epoch {start_epoch - 1}.", flush=True)
            return

        #filter out rows where the target is NaN
        valid_mask = y.notna().to_numpy()
        if not valid_mask.all():
            print(f"[meta head] dropping {(~valid_mask).sum():,} rows with NaN targets from meta training.", flush=True)
            X = X.iloc[valid_mask]
            y = y.iloc[valid_mask]
            
        if val_X is not None and val_y is not None:
            val_valid_mask = val_y.notna().to_numpy()
            if not val_valid_mask.all():
                val_X = val_X.iloc[val_valid_mask]
                val_y = val_y.iloc[val_valid_mask]

        print("[meta head] pre-computing base model predictions for meta training set...", flush=True)
        self.eval()
        with torch.no_grad():
            X_meta = self._encode_meta_features(X)
            base_preds = []
            for (name, model), features in zip(self.ensemble_models, self.features):
                t0 = time.time()
                preds = model.predict(X[features])
                base_preds.append(torch.from_numpy(np.asarray(preds, dtype=np.float32)))
                print(f"[meta head] predicted with base model '{name}' in {time.time() - t0:.1f}s", flush=True)
            base_preds = torch.stack(base_preds, dim=1).to(self.device)

        y_tensor = torch.from_numpy(y.astype(np.float32).to_numpy()).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_meta, base_preds, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW([
            {"params": self.head.parameters(), "weight_decay": weight_decay_head},
            {"params": self.mlp.parameters(), "weight_decay": weight_decay_mlp},
        ], lr=lr)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            print(f"[meta head] restored optimizer state, resuming at epoch {start_epoch}/{epochs}", flush=True)

        def entropy_penalty(weights: torch.Tensor) -> torch.Tensor:
            eps = 1e-8
            entropy = -(weights * (weights + eps).log()).sum(dim=1)
            return -entropy.mean()

        print(f"[meta head] training epochs {start_epoch}..{epochs} "
              f"({len(dataset):,} rows, batch_size={batch_size})", flush=True)

        for epoch in range(start_epoch - 1, epochs):
            epoch_t0 = time.time()
            self.train()
            epoch_loss = 0.0
            epoch_mse = 0.0

            for X_batch, preds_batch, y_batch in loader:
                optimizer.zero_grad()
                weights = torch.softmax(self.mlp(X_batch), dim=1)
                ensemble_pred = (weights * preds_batch).sum(dim=1)
                mse_loss = F.mse_loss(ensemble_pred, y_batch)
                reg_loss = entropy_coef * entropy_penalty(weights)
                loss = mse_loss + reg_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.shape[0]
                epoch_mse += mse_loss.item() * X_batch.shape[0]

            avg_loss = epoch_loss / len(dataset)
            avg_mse = epoch_mse / len(dataset)
            completed_epoch = epoch + 1
            epoch_elapsed = time.time() - epoch_t0

            if checkpoint_dir and (completed_epoch % checkpoint_every == 0 or completed_epoch == epochs):
                metadata = {
                    "stage": "meta",
                    "epoch": completed_epoch,
                    "lr": lr,
                    "weight_decay_head": weight_decay_head,
                    "weight_decay_mlp": weight_decay_mlp,
                    "optimizer_state": optimizer.state_dict(),
                }
                path = checkpoint_meta_path(checkpoint_dir, completed_epoch)
                save_checkpoint(self, path, metadata)
                save_checkpoint_latest(self, checkpoint_dir, metadata)

            if val_X is not None and val_y is not None:
                self.eval()
                with torch.no_grad():
                    val_embeddings = self._encode_meta_features(val_X)
                    val_weights = torch.softmax(self.mlp(val_embeddings), dim=1)
                    val_base_preds = []
                    for (_, model), features in zip(self.ensemble_models, self.features):
                        val_base_preds.append(torch.from_numpy(np.asarray(model.predict(val_X[features].to_numpy()), dtype=np.float32)))
                    val_base_preds = torch.stack(val_base_preds, dim=1).to(self.device)
                    val_pred = (val_weights * val_base_preds).sum(dim=1)
                    val_mse = F.mse_loss(val_pred, torch.from_numpy(val_y.astype(np.float32).to_numpy()).to(self.device))
                print(f"[meta head] epoch={completed_epoch}/{epochs} "
                      f"train_mse={avg_mse:.6f} val_mse={val_mse:.6f} ({epoch_elapsed:.1f}s)", flush=True)
            else:
                print(f"[meta head] epoch={completed_epoch}/{epochs} "
                      f"train_mse={avg_mse:.6f} ({epoch_elapsed:.1f}s)", flush=True)

    def fit(
        self,
        train: pd.DataFrame,
        ensemble_features: Union[List[str], List[List[str]]],
        meta_features: Optional[List[str]] = None,
        targets: Optional[Union[List[str], str]] = None,
        val: Optional[pd.DataFrame] = None,
        base_train: Optional[pd.DataFrame] = None,
        callbacks: Optional[List[Callable]] = None,
        n_jobs: int = -1,
        head_hidden_dim: int = 64,
        head_epochs: int = 20,
        head_lr: float = 1e-3,
        head_batch_size: int = 4096,
        entropy_coef: float = 0.02,
        embed_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        weight_decay_head: float = 1e-2,
        weight_decay_mlp: float = 1e-4,
        checkpoint_dir: Optional[str] = None,
        resume: bool = False,
        checkpoint_every: int = 1,
        base_checkpoint_every_iters: int = 250,
        use_cuda: bool = False,
    ) -> "DynamicEnsemble":
        # `train` is what the meta head trains on (always the full frame you
        # pass in). `base_train`, if given, is what the base LightGBM models
        # train on instead — e.g. `train` minus a small early-stopping
        # holdout slice. This lets the base models early-stop against `val`
        # while the meta head still learns from every row of `train`.
        base_fit_data = base_train if base_train is not None else train

        fit_t0 = time.time()
        n_models = len(self.ensemble_models)
        if n_models == 0:
            raise RuntimeError("No base models added. Call add_model() before fit().")

        print(f"[fit] starting. n_models={n_models} checkpoint_dir={checkpoint_dir} resume={resume} "
              f"base_train_rows={len(base_fit_data):,} meta_train_rows={len(train):,}", flush=True)

        if targets is None:
            targets = ["target"] * n_models
        elif isinstance(targets, str):
            targets = [targets] * n_models

        if isinstance(ensemble_features[0], str):
            per_model_features = [ensemble_features] * n_models
        else:
            per_model_features = list(ensemble_features)

        self.features = per_model_features

        if meta_features is not None:
            self.meta_features = meta_features
        else:
            self.meta_features = sorted({f for subset in self.features for f in subset})

        checkpoint_loaded = False
        resume_epoch = 1
        optimizer_state = None
        checkpoint_metadata = None
        resume_base_model_idx = 0

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if resume:
                latest_path = checkpoint_latest_path(checkpoint_dir)
                base_path = checkpoint_base_path(checkpoint_dir)
                checkpoint_file = None
                if os.path.exists(latest_path):
                    checkpoint_file = latest_path
                elif os.path.exists(base_path):
                    checkpoint_file = base_path

                if checkpoint_file is not None:
                    print(f"[resume] loading checkpoint: {checkpoint_file}", flush=True)
                    checkpoint_data = load_checkpoint(checkpoint_file)
                    self.__dict__.update(checkpoint_data["ensemble"].__dict__)
                    checkpoint_metadata = checkpoint_data["metadata"]
                    checkpoint_loaded = True
                    if checkpoint_metadata.get("stage") == "base":
                        resume_base_model_idx = checkpoint_metadata.get("models_trained", 0)
                        resume_epoch = 1
                        optimizer_state = None
                        print(f"[resume] resuming base model training: "
                              f"{resume_base_model_idx}/{checkpoint_metadata.get('total_models', n_models)} "
                              f"fully complete. Checking for a further in-progress partial model...", flush=True)
                    elif checkpoint_metadata.get("stage") == "meta":
                        resume_epoch = checkpoint_metadata.get("epoch", 0) + 1
                        optimizer_state = checkpoint_metadata.get("optimizer_state")
                        print(f"[resume] resuming meta head training from epoch {resume_epoch}", flush=True)
                    else:
                        resume_epoch = 1
                        optimizer_state = None
                else:
                    print("[resume] --resume was set but no checkpoint found; starting fresh.", flush=True)

        if not checkpoint_loaded:
            self.features = per_model_features
            if meta_features is not None:
                self.meta_features = meta_features
            else:
                self.meta_features = sorted({f for subset in self.features for f in subset})

        if self.head is None or self.mlp is None:
            print("[fit] initializing meta head.", flush=True)
            self._build_meta_head(
                len(self.meta_features) if self.meta_features is not None else len(per_model_features[0]),
                n_models,
                hidden_dim=head_hidden_dim,
                embed_dropout=embed_dropout,
                hidden_dropout=hidden_dropout,
            )

            start_model_idx = resume_base_model_idx if checkpoint_loaded else 0
            if start_model_idx > 0:
                print(f"[base models] skipping {start_model_idx} already-fully-trained models, "
                      f"starting from model {start_model_idx + 1}/{n_models}", flush=True)

            max_concurrent_models = 2
            if n_jobs == 1:
                max_concurrent_models = 1
            elif n_jobs > 1:
                max_concurrent_models = min(max_concurrent_models, n_jobs)
            else:
                max_concurrent_models = 1

            print(f"[base models] launching with max_concurrent_models={max_concurrent_models} "
                  f"(n_jobs={n_jobs}), base_checkpoint_every_iters={base_checkpoint_every_iters}", flush=True)

            with ThreadPoolExecutor(max_workers=max_concurrent_models) as executor:
                futures = {}
                for i in range(start_model_idx, n_models):
                    name, model = self.ensemble_models[i]
                    features = self.features[i]
                    target_col = targets[i]
                    print(f"[base models] submitting model {i+1}/{n_models}: {name} "
                          f"({len(features)} features, target={target_col})", flush=True)
                    future = executor.submit(
                        fit_single_resumable, name, model, base_fit_data, features, target_col, val,
                        callbacks, checkpoint_dir, base_checkpoint_every_iters,
                    )
                    futures[future] = (i, name)

                models_completed = start_model_idx
                print(f"[base models] waiting for {len(futures)} model(s)...", flush=True)
                for future in as_completed(futures):
                    i, name = futures[future]
                    fitted_model = future.result()
                    self.ensemble_models[i] = (name, fitted_model)
                    models_completed += 1

                    print(f"[base models] model {i+1}/{n_models} ('{name}') complete. "
                          f"progress: {models_completed}/{n_models}", flush=True)

                    if checkpoint_dir is not None:
                        metadata = {
                            "stage": "base",
                            "models_trained": models_completed,
                            "total_models": n_models,
                        }
                        save_checkpoint(self, checkpoint_base_path(checkpoint_dir), metadata)
                        save_checkpoint_latest(self, checkpoint_dir, metadata)
                        print(f"[base models] full ensemble checkpoint saved "
                              f"({models_completed}/{n_models} models)", flush=True)

            print(f"[base models] ALL {n_models} models trained in "
                  f"{(time.time() - fit_t0)/60:.1f} min.", flush=True)

        # Move the meta head (embeddings + MLP) onto GPU only if requested —
        # off by default. Also needed after a checkpoint load, since a
        # resumed head/mlp might not be on the intended device.
        self.set_device(use_cuda)

        meta_target = targets[0]
        val_X = val if val is not None and meta_target in val.columns else None
        val_y = val[meta_target] if val is not None and meta_target in val.columns else None
        if base_train is not None and val is not None:
            # In this configuration `val` is the early-stopping holdout carved
            # out of `train`, and the meta head trains on all of `train` —
            # so the "val_mse" the meta head prints below is a soft sanity
            # check, not a clean held-out score (those rows were seen by the
            # meta head, just not by the base trees).
            print("[meta head] note: val set overlaps with meta-head training "
                  "data (it's the early-stopping holdout, not a true holdout "
                  "for the meta head) — treat val_mse as a diagnostic only.", flush=True)

        self._train_meta_head(
            train,
            train[meta_target],
            val_X,
            val_y,
            epochs=head_epochs,
            lr=head_lr,
            batch_size=head_batch_size,
            entropy_coef=entropy_coef,
            weight_decay_head=weight_decay_head,
            weight_decay_mlp=weight_decay_mlp,
            start_epoch=resume_epoch,
            optimizer_state=optimizer_state,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
        )
        print(f"[fit] complete in {(time.time() - fit_t0)/60:.1f} min total.", flush=True)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.ensemble_models:
            raise RuntimeError("No models in ensemble. Call add_model() first.")

        self.eval()
        with torch.no_grad():
            base_preds = []
            for (_, model), features in zip(self.ensemble_models, self.features):
                base_preds.append(torch.from_numpy(np.asarray(model.predict(X[features]), dtype=np.float32)))
            base_preds = torch.stack(base_preds, dim=1).to(self.device)
            weights = self._compute_weights(X)
            ensemble_pred = (weights * base_preds).sum(dim=1)
            return ensemble_pred.cpu().numpy()


def get_checkpoint_dir(model_name: str, job_id: Optional[str] = None, root_dir: str = "./models") -> str:
    if job_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", "local")
    checkpoint_dir = os.path.join(root_dir, model_name, str(job_id))
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def checkpoint_latest_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "checkpoint_latest.pkl")


def checkpoint_base_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "checkpoint_base_models.pkl")


def checkpoint_meta_path(checkpoint_dir: str, epoch: int) -> str:
    return os.path.join(checkpoint_dir, f"checkpoint_meta_epoch_{epoch}.pkl")


def save_checkpoint(ensemble: DynamicEnsemble, path: str, metadata: dict) -> None:
    t0 = time.time()
    with open(path, "wb") as f:
        cloudpickle.dump({"ensemble": ensemble, "metadata": metadata}, f)
    print(f"[checkpoint] saved {path} ({time.time() - t0:.1f}s)", flush=True)


def save_checkpoint_latest(ensemble: DynamicEnsemble, checkpoint_dir: str, metadata: dict) -> None:
    save_checkpoint(ensemble, checkpoint_latest_path(checkpoint_dir), metadata)


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or resume the dynamic ensemble with checkpointing.")
    parser.add_argument("--mode", choices=["prelim", "final"], default="prelim", help="Training mode.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID"), help="SLURM job ID or custom checkpoint subfolder.")
    parser.add_argument("--n-jobs", type=int, default=2, help="Concurrent base models trained in parallel (max 2).")
    parser.add_argument("--checkpoint-every-iters", type=int, default=250, help="Save a partial booster every N boosting rounds.")
    parser.add_argument("--early-stopping-holdout-frac", type=float, default=0.03,
                         help="Fraction of eras (final mode only) held out purely for early stopping. 0 disables early stopping in final mode.")
    parser.add_argument("--use-cuda", action="store_true",
                         help="Train the meta head (embeddings + MLP) on GPU. Off by default — "
                              "the meta head is small enough that CPU is typically as fast or faster.")
    return parser.parse_args()


def build_ensemble(config: Config, params: dict, feature_sets: dict, ensemble_features: list[str]) -> tuple[DynamicEnsemble, list[list[str]], list[str], list[str]]:
    feature_group_names = [
        'intelligence', 'charisma', 'strength', 'dexterity', 'constitution',
        'wisdom', 'agility', 'serenity', 'sunshine', 'rain', 'midnight', 'faith'
    ]

    loaded_features = set(ensemble_features)
    per_model_features: list[list[str]] = []
    for g in feature_group_names:
        group_feats = [f for f in feature_sets[g] if f in loaded_features]
        per_model_features.append(group_feats)

    global_targets = ['target_ender_20', 'target_ender_60', 'target_jasper_20', 'target_jasper_60']
    for _ in global_targets:
        per_model_features.append(ensemble_features)

    ensemble = DynamicEnsemble(embedding_dim=2)
    for group_name in feature_group_names:
        ensemble.add_model(group_name, lgb.LGBMRegressor(**params))
    for target_name in global_targets:
        ensemble.add_model(target_name, lgb.LGBMRegressor(**params))

    return ensemble, per_model_features, global_targets, feature_group_names


def run_training(
    mode: str,
    resume: bool = False,
    job_id: Optional[str] = None,
    n_jobs: int = 2,
    checkpoint_every_iters: int = 250,
    early_stopping_holdout_frac: float = 0.0,
    use_cuda: bool = False,
) -> None:
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[run] mode={mode} resume={resume} job_id={job_id} n_jobs={n_jobs}", flush=True)

    config = Config(model_name='dynamic_ensemble', feature_set="all")
    print("[run] loading train split...", flush=True)
    train, ensemble_features = load_dataset(config, split="train")
    print(f"[run] train loaded: {len(train):,} rows, {len(ensemble_features)} features", flush=True)
    print("[run] loading validation split...", flush=True)
    val, _ = load_dataset(config, split="validation")
    print(f"[run] validation loaded: {len(val):,} rows", flush=True)

    train[ensemble_features] = train[ensemble_features].astype(np.int8)
    val[ensemble_features] = val[ensemble_features].astype(np.int8)

    checkpoint_dir = get_checkpoint_dir(config.model_name, job_id)
    print(f"[run] checkpoint_dir={checkpoint_dir}", flush=True)

    # Use SLURM's allocated CPU count (respects cgroup limits) rather than
    # os.cpu_count(), which can report the whole node's core count.
    try:
        allocated_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        allocated_cpus = os.cpu_count() or 1
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", allocated_cpus))
    cpu_budget = min(allocated_cpus, slurm_cpus)
    # Split the CPU budget across the concurrently-training models so we
    # don't oversubscribe.
    threads_per_model = max(1, cpu_budget // max(1, n_jobs))
    print(f"[run] allocated_cpus={allocated_cpus} slurm_cpus={slurm_cpus} "
          f"-> threads_per_model={threads_per_model} (n_jobs={n_jobs})", flush=True)

    params = {
        "num_threads": threads_per_model,
        "num_leaves": 64,
        "colsample_bytree": 0.1,
        "subsample": 0.7,
        "subsample_freq": 1,
        "learning_rate": 0.001,
        "n_estimators": 10_000,
        "min_child_samples": 1000,
        "verbose": -1,
        "reg_lambda": 5.0,
        "seed": seed,
        "deterministic": True,
        "bin_construct_sample_cnt": 200_000,
        "free_raw_data": True,
    }

    feature_sets = load_feature_groups()
    meta_feats_curated = feature_sets.get("small", list(set(ensemble_features))[:40])
    meta_feats_curated = [f for f in meta_feats_curated if f in ensemble_features]

    ensemble, per_model_features, global_targets, _ = build_ensemble(config, params, feature_sets, ensemble_features)

    base_train_data = None
    if mode == "prelim":
        fit_data = train
        val_data = val
        callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
        last_train_era = int(train["era"].unique()[-1])
    else:
        train_val = pd.concat([train, val], ignore_index=True)
        fit_data = train_val
        last_train_era = int(train_val["era"].unique()[-1])
        del train
        del val
        gc.collect()

        if early_stopping_holdout_frac and early_stopping_holdout_frac > 0:
            # Carve a small chronological slice out purely so the base trees
            # know when to stop — without this, `final` mode has no early
            # stopping and every model runs the full n_estimators (10,000 at
            # lr=0.001), which is by far the slowest part of the pipeline.
            # The meta head below still trains on the FULL fit_data (all of
            # train+validation) — see DynamicEnsemble.fit()'s base_train arg.
            base_train_data, val_data = carve_era_holdout(fit_data, early_stopping_holdout_frac)
            callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
        else:
            val_data = None
            callbacks = None
            print("[run] early_stopping_holdout_frac=0 — base models will run "
                  "the full n_estimators with no early stopping (slow).", flush=True)

    print("[run] starting ensemble.fit()...", flush=True)
    ensemble.fit(
        fit_data,
        ensemble_features=per_model_features,
        meta_features=meta_feats_curated,
        targets=["target"] * len(per_model_features[:12]) + global_targets,
        val=val_data,
        base_train=base_train_data,
        callbacks=callbacks,
        n_jobs=n_jobs,
        head_hidden_dim=64,
        head_epochs=20,
        head_lr=1e-3,
        head_batch_size=1024,
        entropy_coef=0.02,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        checkpoint_every=1,
        base_checkpoint_every_iters=checkpoint_every_iters,
        use_cuda=use_cuda,
    )

    print("[run] saving final model...", flush=True)
    save_model(ensemble, config, last_train_era=last_train_era)
    print("[run] done.", flush=True)


def train_all(
    resume: bool = False,
    job_id: Optional[str] = None,
    n_jobs: int = 2,
    checkpoint_every_iters: int = 250,
    early_stopping_holdout_frac: float = 0.03,
    use_cuda: bool = False,
) -> None:
    """Train the submission-ready model on train + validation combined.

    This is what you want before generating tournament predictions to
    actually submit — it maximizes the data available to the model, same as
    `--mode final`. It's broken out into its own function (rather than only
    a --mode flag) so it's unambiguous at the call site which run produces
    your submission model vs. a prelim/dev run.

    Early stopping: a small chronological slice of eras
    (`early_stopping_holdout_frac`, default 3%) is carved out purely to
    tell each base LightGBM model when to stop boosting. That slice is
    excluded from the base trees' .fit() calls but IS still included in the
    meta head's training data — no data is discarded, it's just used
    differently by the two stages. Pass early_stopping_holdout_frac=0.0 to
    disable this and go back to training every base tree on 100% of the
    data for the full n_estimators (slower, and what the old code did).

    use_cuda: train the meta head on GPU instead of CPU. Off by default.
    """
    run_training(
        mode="final",
        resume=resume,
        job_id=job_id,
        n_jobs=n_jobs,
        checkpoint_every_iters=checkpoint_every_iters,
        early_stopping_holdout_frac=early_stopping_holdout_frac,
        use_cuda=use_cuda,
    )


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "final":
        train_all(
            resume=args.resume,
            job_id=args.job_id,
            n_jobs=args.n_jobs,
            checkpoint_every_iters=args.checkpoint_every_iters,
            early_stopping_holdout_frac=args.early_stopping_holdout_frac,
            use_cuda=args.use_cuda,
        )
    else:
        run_training(
            mode="prelim",
            resume=args.resume,
            job_id=args.job_id,
            n_jobs=args.n_jobs,
            checkpoint_every_iters=args.checkpoint_every_iters,
            use_cuda=args.use_cuda,
        )