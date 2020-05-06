from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config
import os
from datetime import timezone, date, timedelta
from pandas.tseries import offsets

config = Config('GeometryUtilityConfig')

config.core_poi_apr_data_path = os.path.join(global_config.data_base_dir, "core_places/CoreApr2020Release-CORE_POI-2020_03-2020-04-07/")

config.svi_data_us_county_data_path = os.path.join(global_config.data_base_dir, "SVI2018_US_COUNTY/")


sys.modules[__name__] = config
