from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config
import os

config = Config('Config for CovidCountyDataset')

config.labels_key = 'labels'
config.labels_class_boundaries = [-100, -10, 0, 10, 100]
config.num_classes = len(config.labels_class_boundaries) + 1

config.get_cached_tensors_path = \
    lambda s, e: os.path.join(global_config.data_save_dir, f'tensors_{str(s)}_{str(e)}')

config.num_features = 5

sys.modules[__name__] = config
