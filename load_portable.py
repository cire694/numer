# load_portable.py — run in the 3.12 env
import json
import torch
import lightgbm as lgb
from train_models.dynamic_ensemble import DynamicEnsemble  # your class def, not pickled

def load_portable_ensemble(portable_dir: str) -> DynamicEnsemble:
    with open(f"{portable_dir}/metadata.json") as f:
        meta = json.load(f)

    ensemble = DynamicEnsemble(embedding_dim=meta["embedding_dim"])

    # Rebuild base models as raw lgb.Booster objects — Booster.predict() has
    # the same call signature your DynamicEnsemble.predict() already uses,
    # so no sklearn wrapper is needed for inference-only usage.
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

if __name__ == "__main__":
    ensemble = load_portable_ensemble("models/dynamic_ensemble_20260728_225421_portable")
    print("Portable load OK:", ensemble)