"""
Self-contained Model Upload builder.

CRITICAL: DynamicEnsemble is defined directly in this file (not imported
from train_models.dynamic_ensemble) so that cloudpickle serializes the
class BY VALUE. If it were imported from a real module, cloudpickle would
instead store a reference like "import train_models.dynamic_ensemble" —
which fails on Numerai's sandbox since that codebase doesn't exist there
("No module named 'train_models'").

Run under python 3.12 (conda activate numer_upload), after
export_portable.py has already produced the portable artifacts.

Usage:
    python predict_upload.py --portable-dir models/dynamic_ensemble_XXXX_portable \
        --download-name predict_dynamic_ensemble
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import cloudpickle
from typing import List, Optional, Any


# ── Inference-only, dependency-light copy of DynamicEnsemble ────────────
# No `data.py` / `numerapi` imports anywhere below — only torch/numpy/pandas,
# all of which are in Numerai's pinned py3.12 requirements.
class DynamicEnsemble(nn.Module):
    def __init__(self, embedding_dim: int = 2) -> None:
        super().__init__()
        self.ensemble_models: List[tuple] = []
        self.features: Optional[List[List[str]]] = None
        self.meta_features: Optional[List[str]] = None
        self.embedding_dim = embedding_dim
        self.head: Optional[nn.Embedding] = None
        self.mlp: Optional[nn.Module] = None
        self.embed_drop: Optional[nn.Dropout] = None
        self.device: torch.device = torch.device("cpu")
        self.register_buffer("offsets", torch.empty(0, dtype=torch.long))

    def add_model(self, name: str, model: Any) -> "DynamicEnsemble":
        self.ensemble_models.append((name, model))
        return self

    def _build_meta_head(self, num_meta_features, num_models, hidden_dim=64, embed_dropout=0.0):
        self.head = nn.Embedding(5 * num_meta_features, self.embedding_dim)
        self.register_buffer("offsets", torch.arange(0, num_meta_features * 5, 5, dtype=torch.long))
        self.embed_drop = nn.Dropout(embed_dropout) if embed_dropout > 0.0 else None
        self.mlp = nn.Sequential(
            nn.Linear(num_meta_features * self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_models),
        )

    def _meta_feature_indices(self, X: pd.DataFrame) -> torch.LongTensor:
        x = X.loc[:, self.meta_features].fillna(2).astype(np.int64).to_numpy(copy=True)
        indices = torch.from_numpy(x).to(self.offsets.device)
        return indices + self.offsets

    def _encode_meta_features(self, X: pd.DataFrame) -> torch.Tensor:
        indices = self._meta_feature_indices(X)
        embeddings = self.head(indices)
        if self.embed_drop is not None:
            embeddings = self.embed_drop(embeddings)
        return embeddings.view(embeddings.shape[0], -1)

    def _compute_weights(self, X: pd.DataFrame) -> torch.Tensor:
        embeddings = self._encode_meta_features(X)
        logits = self.mlp(embeddings)
        return torch.softmax(logits, dim=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            base_preds = []
            for (_, model), features in zip(self.ensemble_models, self.features):
                # model is a raw lgb.Booster here (see load_portable_ensemble below)
                preds = model.predict(X[features].to_numpy())
                base_preds.append(torch.from_numpy(np.asarray(preds, dtype=np.float32)))
            base_preds = torch.stack(base_preds, dim=1).to(self.device)
            weights = self._compute_weights(X)
            ensemble_pred = (weights * base_preds).sum(dim=1)
            return ensemble_pred.cpu().numpy()


def load_portable_ensemble(portable_dir: str) -> DynamicEnsemble:
    with open(f"{portable_dir}/metadata.json") as f:
        meta = json.load(f)

    ensemble = DynamicEnsemble(embedding_dim=meta["embedding_dim"])

    for name in meta["model_names"]:
        booster = lgb.Booster(model_file=f"{portable_dir}/booster_{name}.txt")
        ensemble.add_model(name, booster)

    ensemble.features = meta["features"]
    ensemble.meta_features = meta["meta_features"]

    ensemble._build_meta_head(
        meta["num_meta_features"],
        meta["num_models"],
        hidden_dim=meta["hidden_dim"],
        embed_dropout=meta["embed_dropout"],
    )

    state = torch.load(f"{portable_dir}/meta_head.pt", map_location="cpu")
    ensemble.head.load_state_dict(state["head_state_dict"])
    ensemble.mlp.load_state_dict(state["mlp_state_dict"])
    if state["embed_drop_state_dict"] is not None and ensemble.embed_drop is not None:
        ensemble.embed_drop.load_state_dict(state["embed_drop_state_dict"])
    ensemble.offsets = state["offsets"]

    return ensemble


def build_upload_pickle(portable_dir: str, download_name: str) -> None:
    model = load_portable_ensemble(portable_dir)

    def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        preds = model.predict(live_features)
        return pd.Series(preds, index=live_features.index).to_frame("prediction")

    payload = cloudpickle.dumps(predict)
    out_path = f"{download_name}.pkl"
    with open(out_path, "wb") as f:
        f.write(payload)
    print(f"Wrote {out_path} ({len(payload) / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable-dir", required=True)
    parser.add_argument("--download-name", default="predict_dynamic_ensemble")
    args = parser.parse_args()
    build_upload_pickle(args.portable_dir, args.download_name)