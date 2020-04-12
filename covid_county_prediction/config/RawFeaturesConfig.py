from covid_county_prediction.config.base_config import Config
import sys
from enum import Enum, auto

config = Config('RawFeatures config parameters')

class FeatureType(Enum):
    CONSTANTS                   = auto() # time independent
    TIME_DEPENDENT              = auto() 
    COUNTY_WISE_TIME_DEPENDENT  = auto()

config.feature_type = FeatureType

sys.modules[__name__] = config