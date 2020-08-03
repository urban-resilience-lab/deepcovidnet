from deepcovidnet.config.base_config import Config
import deepcovidnet.config.global_config as global_config
import os, sys


config = Config('Global config parameters')

config.training_mean_std_file = os.path.join(
                                    global_config.data_save_dir,
                                    'train_mean_std.pickle'
                                )

config.get_spatial_csv = lambda dt : os.path.join(global_config.data_save_dir, f'viz/spatial_{dt}.csv')

sys.modules[__name__] = config
