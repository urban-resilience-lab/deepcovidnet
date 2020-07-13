from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config
import os
import logging
from datetime import date


config = Config('Config for CovidCountyDataset')

config.labels_key = 'labels'
config.labels_class_boundaries = [1, 13, 93]  # 0.31, 0.67, 0.9: 4/5 - 6/29
# [2, 11, 80]  # 0.33, 0.67, 0.9 percentiles 1/28 - 6/7

config.label_to_str_range = {}
config.label_to_range = {}

for i in range(len(config.labels_class_boundaries) + 1):
    low, high = 0, None
    if i > 0:
        low = config.labels_class_boundaries[i - 1] + 1
    if i < len(config.labels_class_boundaries):
        high = config.labels_class_boundaries[i]
    config.label_to_str_range[i] = f'{low}{"+" if high is None else f"-{high}"}'
    config.label_to_range[i] = (low, high)

assert global_config.data_start_date == date(2020, 4, 5), 'change label_class_boundaries accordingly before use or explicitly comment this out'
assert global_config.data_end_date == date(2020, 6, 29), 'change label_class_boundaries accordingly or explicitly comment this out'

config.num_classifiers = len(config.labels_class_boundaries)
config.num_classes = len(config.labels_class_boundaries) + 1

tensor_dir = os.path.join(global_config.data_save_dir, 'tensors/')
if not os.path.exists(tensor_dir):
    os.mkdir(tensor_dir)


def get_cached_tensors_path(s, e):
    base_file = f'tensors_{str(s)}_{str(e)}.pt'
    loc = os.path.join(tensor_dir, base_file)
    mem_loc = os.path.join('/dev/shm/', base_file)
    if os.path.exists(mem_loc):
        logging.info(f'Leading {base_file} from memory')
        return mem_loc
    return loc


config.get_cached_tensors_path = get_cached_tensors_path

config.num_features = 8

sys.modules[__name__] = config
