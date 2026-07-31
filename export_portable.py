"""
export_portable.py — run in the 3.14 env (numer), where the original
cloudpickled ensemble still loads fine.
This is only for dynamic ensemble

Exports the ensemble to version-agnostic files:
  - one LightGBM .txt per base model (native format, not a Python pickle)
  - meta head (embeddings + MLP) via plain torch.save
  - metadata.json describing how to reconstruct the DynamicEnsemble shell

Usage:
    python export_portable.py --model-path models/dynamic_ensemble_XXXX.pkl \
        --out-dir models/dynamic_ensemble_XXXX_portable
"""
import argparse
import os
import json
import torch
from utils import load_model


def export_portable(model_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ensemble = load_model(model_path)

    model_names = []
    for name, model in ensemble.ensemble_models:
        booster = model.booster_ if hasattr(model, "booster_") else model
        path = os.path.join(out_dir, f"booster_{name}.txt")
        booster.save_model(path)
        model_names.append(name)
        print(f"saved booster: {name}")

    torch.save({
        "head_state_dict": ensemble.head.state_dict(),
        "mlp_state_dict": ensemble.mlp.state_dict(),
        "embed_drop_state_dict": ensemble.embed_drop.state_dict() if ensemble.embed_drop is not None else None,
        "offsets": ensemble.offsets,
    }, os.path.join(out_dir, "meta_head.pt"))

    metadata = {
        "model_names": model_names,
        "features": ensemble.features,
        "meta_features": ensemble.meta_features,
        "embedding_dim": ensemble.embedding_dim,
        "num_meta_features": len(ensemble.meta_features),
        "num_models": len(ensemble.ensemble_models),
        "hidden_dim": ensemble.mlp[0].out_features,
        "embed_dropout": ensemble.embed_drop.p if ensemble.embed_drop is not None else 0.0,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Portable export complete: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the original .pkl")
    parser.add_argument("--out-dir", required=True, help="Directory to write portable files into")
    args = parser.parse_args()
    export_portable(args.model_path, args.out_dir)