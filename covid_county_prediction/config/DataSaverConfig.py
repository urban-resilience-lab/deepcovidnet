from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config
import datetime
import os


config = Config('Config for DataSaver')


def get_file_func(root, file_format):
    return lambda d: d.strftime(
            os.path.join(root, file_format)
        )


# census data
config.census_data_root = \
    os.path.join(global_config.data_save_dir, 'sg_census_data')

config.census_data_path = \
    os.path.join(config.census_data_root, 'sg_census_data.csv')

# monthly data
config.sg_patterns_monthly_root = \
    os.path.join(global_config.data_save_dir, 'monthly_patterns')

config.sg_patterns_monthly_file_format = \
    '%Y-%m-%d-monthly-patterns.csv'

config.get_sg_patterns_monthly_file = \
    get_file_func(config.sg_patterns_monthly_root, config.census_data_path)

# social distancing data
config.sg_social_distancing_root = \
    os.path.join(global_config.data_save_dir, 'social_distancing')

config.sg_social_distancing_file_format = \
    '%Y-%m-%d-social-distancing.csv'

config.get_sg_social_distancing_file = \
    get_file_func(config.sg_social_distancing_root,
                  config.sg_social_distancing_file_format)

# weather data
config.weather_root = \
    os.path.join(global_config.data_save_dir, 'weather_data')

config.weather_file_format = \
    '%Y-%m-%d-weather.csv'

config.get_weather_file = \
    get_file_func(config.weather_root, config.weather_file_format)

# sg mobility data
config.sg_mobility_root = \
    os.path.join(global_config.data_save_dir, 'sg_mobility')

config.sg_mobility_file_format = \
    '%Y-%m-%d-mobility.csv'

config.get_sg_mobility_file = \
    get_file_func(config.weather_root, config.weather_file_format)

sys.modules[__name__] = config
