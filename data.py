from typing import Tuple, List
import os, json
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI
from config import Config

def get_target_cols(pq_path: str) -> List[str]:
    schema = pq.read_schema(pq_path)
    return [name for name in schema.names if name.startswith("target")]

def get_features(config: Config, split: str = "train") -> List[str]:
    """Get feature list for a given config and split without loading data.

    Args:
        config: Configuration object with feature_set and data_version.
        split: Data split ('train', 'validation', or 'live').

    Returns:
        List of feature column names for the configured feature_set.
    """
    dest_folder = os.path.join(config.data_dir, split)
    os.makedirs(dest_folder, exist_ok=True)

    json_out = os.path.join(dest_folder, "features.json")
    if not os.path.exists(json_out):
        print(f"{json_out} does not exist, downloading features.json")
        napi = NumerAPI()
        napi.download_dataset(f"{config.data_version}/features.json", json_out)
    
    features = json.load(open(json_out))["feature_sets"][config.feature_set]
    return features

def load_dataset(config: Config, split: str = "train") -> Tuple[pd.DataFrame, List[str]]:
    """Load a Numerai dataset split and cache it locally.

    Args:
        config: Configuration object with data settings.
        split: Data split to load ('train', 'validation', or 'live').

    Returns:
        A tuple containing the loaded DataFrame and feature list.
    """
    napi = NumerAPI()
    dest_folder = os.path.join(config.data_dir, split)
    os.makedirs(dest_folder, exist_ok = True)

    json_out = os.path.join(dest_folder, "features.json")
    if not os.path.exists(json_out):
        print(f"{json_out} does not exist, downloading {json_out}/features.json")
        napi.download_dataset(f"{config.data_version}/features.json", json_out)
    features = json.load(open(json_out))["feature_sets"][config.feature_set]

    pq_out = os.path.join(dest_folder, f"{split}.parquet")
    if not os.path.exists(pq_out):
        print(f"{pq_out} does not exist, downloading {pq_out}/features.json")
        napi.download_dataset(f"{config.data_version}/{split}.parquet", pq_out)
    
    cols = ["era"] + get_target_cols(pq_out) + features
    if split == "live":
        cols = ["era"] + features # live data has no target

    df = pd.read_parquet(pq_out, columns=cols)
    return df, features