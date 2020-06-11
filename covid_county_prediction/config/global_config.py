from covid_county_prediction.config.base_config import Config
import sys
import os
from pathlib import Path
from datetime import date
from comet_ml import Experiment


config = Config('Global config parameters')

data_base_dir = '/data'
data_save_dir = '/saved_covid_data'

config.set_static_val('data_base_dir', data_base_dir)
config.set_static_val('data_save_dir', data_save_dir)

config.test_split_pct = 0.2
config.train_split_pct = 0.85 * (1 - config.test_split_pct)
config.val_split_pct = \
    1 - config.test_split_pct - config.train_split_pct

config.data_start_date = date(2020, 1, 21)
config.data_end_date = date(2020, 6, 1)

# comet_ml
comet_api_key = 'q0r1Pj8nOXNDahDBgYDUrnj0x'
comet_project = 'covid-county-prediction'
workspace = 'ankit61'

config.set_static(
    'comet_exp',
    Experiment,
    [comet_api_key, comet_project, workspace],
    break_args=True
)

assert config.test_split_pct + config.train_split_pct + config.val_split_pct

sys.modules[__name__] = config
