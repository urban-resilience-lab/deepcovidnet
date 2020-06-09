from covid_county_prediction.config.base_config import Config
import sys
import pandas as pd

config = Config('general features config parameters')


def get_county_info(county_info_link):
    df = pd.read_html(county_info_link)[0].iloc[:-1].set_index('FIPS')
    duplicates = df.loc[df.index.duplicated()].index
    df = df.drop(duplicates)
    return df


def get_iloc_to_county(county_df):
    ans = []
    for i in range(county_df.shape[0]):
        ans.append(county_df.iloc[i].name)
    return ans


def get_county_name_to_iloc(county_df):
    ans = {}
    for i in range(county_df.shape[0]):
        ans[county_df.iloc[i].name] = i
    return ans

county_info_link = 'https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697'

config.set_static('county_info', get_county_info, county_info_link)
config.set_static('iloc_to_county', get_iloc_to_county, config.county_info)
config.set_static('county_to_iloc', get_county_name_to_iloc, config.county_info)


sys.modules[__name__] = config
