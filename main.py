from config import Config
from pipeline import run

config = Config(model_name="ridge", data_version="v5.2", feature_set="small")
results = run(config)
print(results)