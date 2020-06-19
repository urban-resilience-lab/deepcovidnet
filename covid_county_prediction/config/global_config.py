from covid_county_prediction.config.base_config import Config
import sys
import os
from pathlib import Path
from datetime import date, timedelta


config = Config('Global config parameters')

data_base_dir = '/data'
data_save_dir = '/saved_covid_data'

config.set_static_val('data_base_dir', data_base_dir)
config.set_static_val('data_save_dir', data_save_dir)

config.data_start_date = date(2020, 1, 28)
config.data_end_date = date(2020, 6, 7)

config.train_end_date   = date(2020, 5, 15)
config.val_end_date     = date(2020, 5, 24)

sys.modules[__name__] = config
