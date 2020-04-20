from covid_county_prediction.config.base_config import Config

config = Config('Hyperparameters for DNN model')

config.deep_fm_in_feature_size = 256

sys.modules[__name__] = config