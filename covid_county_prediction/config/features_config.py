from covid_county_prediction.config.base_config import Config
import sys
import pandas as pd

config = Config('general features config parameters')

def get_county_info(county_info_link):
    return pd.read_html(county_info_link)[0].iloc[:-1].set_index('FIPS')

county_info_link = 'https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697'

config.set_static('county_info', get_county_info, county_info_link)

sys.modules[__name__] = config