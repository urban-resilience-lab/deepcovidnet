from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config
import os

config = Config('Config for CovidCountyDataset')

config.labels_key = 'labels'
config.labels_class_boundaries = [-100, -10, 0, 10, 100]
config.num_classes = len(config.labels_class_boundaries) + 1

tensor_dir = os.path.join(global_config.data_save_dir, 'tensors/')
if not os.path.exists(tensor_dir):
    os.mkdir(tensor_dir)


def get_cached_tensors_path(s, e):
    base_file = f'tensors_{str(s)}_{str(e)}.pt'
    loc = os.path.join(tensor_dir, base_file)
    mem_loc = os.path.join('/dev/shm/', base_file)
    if os.path.exists(mem_loc):
        print(mem_loc)
        return mem_loc
    print(loc)
    return loc

config.get_cached_tensors_path = get_cached_tensors_path

config.num_features = 5

sys.modules[__name__] = config
