from covid_county_prediction.config.base_config import Config
import sys

config = Config('Hyperparameters for DNN model')

config.embedding_size               = 256
config.higher_order_features_size   = 256

config.epochs                       = 300

sys.modules[__name__] = config
