from covid_county_prediction.config.base_config import Config
import sys

config = Config('Config for CovidCountyDataset')

config.labels_key = 'labels'
config.labels_class_boundaries = [-100, -10, 0, 10, 100]

sys.modules[__name__] = config