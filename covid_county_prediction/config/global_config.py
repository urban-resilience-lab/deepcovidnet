from covid_county_prediction.config.base_config import Config
import sys
import os
from pathlib import Path

config = Config('Global config parameters')

config.data_base_dir = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, 'data')

sys.modules[__name__] = config