from covid_county_prediction.config.base_config import Config
import sys
import os, getpass
from datetime import datetime
from pathlib import Path

config = Config('BaseRunnerConfig')

config.print_freq               = 200
config.intermittent_output_freq = 5 # Num batches between outputs
config.save_freq                = 5


file_dir = os.path.dirname(os.path.abspath(__file__))

config.get_tensorboard_dir = \
    lambda exp_name: os.path.join(
                        Path(file_dir).parent.parent,
                        'runs',
                        datetime.now().strftime('%b%d_%H-%M') + '_' + exp_name
                    )

config.models_base_dir = os.path.join(Path(file_dir).parent.parent, 'models')
config.min_save_acc = 0.735

if not os.path.exists(config.models_base_dir):
    os.mkdir(config.models_base_dir)

sys.modules[__name__] = config
