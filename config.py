from dataclasses import dataclass, field

@dataclass
class Config: 
    data_version: str = "v5.2"
    feature_set: str = "medium"
    data_dir: str = "./data/"
    model_name: str = "none" # 'xgboost', 'lgbm', 'ridge', or 'linear'
    model_params: dict = field(default_factory=lambda:{
    })
    