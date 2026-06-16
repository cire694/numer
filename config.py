from dataclasses import dataclass, field

@dataclass
class Config: 
    data_version: str = "v5.2"
    feature_set: str = "medium"
    data_dir: str = "./data"
    model_name: str = "xgboost"
    model_params: dict = field(default_factory=lambda:{
        "n_estimators": 2000, 
        "learning_rate": 0.01, 
        "max_depth": 5, 
        "colsample_bytree": 0.1,
    })
    