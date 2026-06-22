from dataclasses import dataclass, field

@dataclass
class Config: 
    data_version: str = "v5.2"
    feature_set: str = "medium"
    data_dir: str = "./data/"
    val_downsample: int = 4
    model_name: str = "none" # see MODEL_REGISTRY in models/__init__.py
    model_params: dict = field(default_factory=lambda:{
    })
    