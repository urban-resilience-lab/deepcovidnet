from deepcovidnet.config.base_config import Config
import sys

config = Config('Config for Hyperparameters')

config.total_trials = 30

sys.modules[__name__] = config
