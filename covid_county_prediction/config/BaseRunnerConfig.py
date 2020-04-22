from covid_county_prediction.config.base_config import Config
import sys
import os, getpass
from datetime import datetime
from pathlib import Path

config = Config('BaseRunnerConfig')

config.print_freq               = 20
config.intermittent_output_freq = 5 # Num batches between outputs
config.save_freq                = 5

config.min_learning_rate        = 0.000001

config.lr_decay_step_size       = 10
config.lr_decay_factor          = 0.9

config.tensorboardx_base_dir = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
config.models_base_dir = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent.parent, 'models')

sys.modules[__name__] = config

