from covid_county_prediction.config.base_config import Config
import sys
import os
from pathlib import Path
from datetime import date, datetime


config = Config('Global config parameters')

data_base_dir = '/data'
data_save_dir = '/saved_covid_data'

config.set_static_val('data_base_dir', data_base_dir)
config.set_static_val('data_save_dir', data_save_dir)

config.data_start_date = date(2020, 4, 5)
config.data_end_date = date(2020, 6, 29)

config.train_end_date   = date(2020, 6, 2)
config.val_end_date     = date(2020, 6, 12)


def get_best_tune_file(exp_name):
    now = datetime.now()
    fl = f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}_{exp_name}.pickle'
    dr = os.path.join(config.data_save_dir, 'tunes')
    if not os.path.exists(dr):
        os.mkdir(dr)
    return os.path.join(dr, fl)


config.get_best_tune_file = get_best_tune_file

sys.modules[__name__] = config
