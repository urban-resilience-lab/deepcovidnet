from covid_county_prediction.config.base_config import Config
import sys

config = Config('Config for CountyWiseTimeDependentFeatures')

config.cross_type = 'CROSS'
config.const_type = 'CONSTANT'
config.types = [config.cross_type, config.const_type]

sys.modules[__name__] = config
