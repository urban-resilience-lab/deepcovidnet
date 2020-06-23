from covid_county_prediction.config.base_config import Config
import covid_county_prediction.config.global_config as global_config
import sys
from datetime import datetime
import os

config = Config('FeatureAnalyzerConfig')


def get_ranks_file(exp):
    n = datetime.now()

    fl = f'rank_{exp}_{str(n.date())}_{n.hour}-{n.minute}.csv'
    dr = os.path.join(global_config.data_save_dir, 'ranks')

    if not os.path.exists(dr):
        os.mkdir(dr)

    return os.path.join(dr, fl)


config.get_ranks_file = get_ranks_file

sys.modules[__name__] = config
