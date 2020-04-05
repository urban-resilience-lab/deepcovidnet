from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config
import os

config = Config('Config for BaseCountyDataset')

#core poi
config.core_poi_csv_prefix = 'core_poi-part'
config.core_poi_path = os.path.join(global_config.data_base_dir, 'CoreRecords-CORE_POI-2019_03-2020-03-25')

#open census
config.sg_open_census_data_path = os.path.join(global_config.data_base_dir, "safegraph_open_census_data/data/")

#poi -> county
config.place_county_cbg_file = os.path.join(global_config.data_base_dir, 'placeCountyCBG.csv')

#labels
config.labels_csv_path = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'

sys.modules[__name__] = config