from pipeline import run
from config import Config
from utils import save_model

config = Config(
    model_name = "lgbm",
    val_downsample=1, #no downsampling
    model_params = {
        "n_estimators": 2000, 
        "learning_rate": 0.01, 
        "max_depth": 5, 
        "num_leaves": 2**5 - 1, 
        "colsample_bytree": 0.1
    }
)

validation, model, last_train_era = run(config)
print(validation)
save_model(model, config, "lgbm", last_train_era=last_train_era)

