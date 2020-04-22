from covid_county_prediction.config.base_config import Config
import sys
import os
from pathlib import Path

config = Config('Global config parameters')

data_base_dir = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, 'data')
data_save_dir = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, 'saved_covid_data')

config.set_static_val('data_base_dir', data_base_dir)
config.set_static_val('data_save_dir', data_save_dir)

sys.modules[__name__] = config

