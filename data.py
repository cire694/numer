import os, json
import pandas as pd
from numerapi import NumerAPI

#split: train, validation, live
def load_dataset(config, split="train"):
    napi = NumerAPI()
    dest_folder = config.data_dir
    os.makedirs(dest_folder, exist_ok = True)

    json_out = os.path.join(dest_folder, "features.json")
    if not os.path.exists(json_out): 
        napi.download_dataset(f"{config.data_version}/features.json", json_out)
    features = json.load(open(json_out))["feature_sets"][config.feature_set]

    pq_out = os.path.join(dest_folder, f"{split}.parquet")
    if not os.path.exists(pq_out):
        napi.download_dataset(f"{config.data_version}/{split}.parquet", pq_out)
    
    cols = ["era", "target"] + features
    if split == "live":
        cols = ["era"] + features # live data has no target

    df = pd.read_parquet(pq_out, columns=cols)
    return df, features