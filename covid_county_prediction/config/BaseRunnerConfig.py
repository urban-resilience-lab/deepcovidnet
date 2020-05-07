from covid_county_prediction.config.base_config import Config
import sys
import os, getpass
from datetime import datetime
from pathlib import Path

config = Config('BaseRunnerConfig')

config.print_freq               = 25
config.intermittent_output_freq = 5 # Num batches between outputs
config.save_freq                = 5


config.tensorboardx_base_dir = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
config.models_base_dir = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, 'models')

if not os.path.exists(config.models_base_dir):
    os.mkdir(config.models_base_dir)

sys.modules[__name__] = config

