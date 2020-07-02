from covid_county_prediction.config.base_config import Config
import covid_county_prediction.config.global_config as global_config
import os, sys


config = Config('Global config parameters')

config.training_mean_std_file = os.path.join(
                                    global_config.data_save_dir,
                                    'train_mean_std.pickle'
                                )

sys.modules[__name__] = config
