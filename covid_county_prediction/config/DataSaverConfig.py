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

config.sg_patterns_monthly_file_format = '%Y-%m-%d-monthly-patterns.csv'

config.get_sg_patterns_monthly_file = \
    get_file_func(config.sg_patterns_monthly_root, config.sg_patterns_monthly_file_format)

config.sg_patterns_monthly_mean_file = \
    os.path.join(config.sg_patterns_monthly_root, 'monthly-patterns-mean.pickle')

config.sg_patterns_monthly_std_file = \
    os.path.join(config.sg_patterns_monthly_root, 'monthly-patterns-std.pickle')

# social distancing data
config.sg_social_distancing_root = \
    os.path.join(global_config.data_save_dir, 'social_distancing')

config.sg_social_distancing_file_format = '%Y-%m-%d-social-distancing.csv'

config.get_sg_social_distancing_file = \
    get_file_func(config.sg_social_distancing_root,
                  config.sg_social_distancing_file_format)

config.sg_social_distancing_mean_file = \
    os.path.join(config.sg_social_distancing_root, 'social-distancing-mean.pickle')

config.sg_social_distancing_std_file = \
    os.path.join(config.sg_social_distancing_root, 'social-distancing-std.pickle')

# weather data
config.weather_root = \
    os.path.join(global_config.data_save_dir, 'weather_data')

config.weather_file_format = '%Y-%m-%d-weather.csv'

config.get_weather_file = \
    get_file_func(config.weather_root, config.weather_file_format)

# num cases data
config.num_cases_root = \
    os.path.join(global_config.data_save_dir, 'num_cases')

config.num_cases_file_format = '%Y-%m-%d-num-cases.csv'

config.get_num_cases_file = \
    get_file_func(config.num_cases_root, config.num_cases_file_format)

config.num_cases_mean_file = \
    os.path.join(config.num_cases_root, 'num-cases-mean.pickle')

config.num_cases_std_file = \
    os.path.join(config.num_cases_root, 'num-cases-std.pickle')

# countywise cumulative cases data
config.countywise_cumulative_cases_root = \
    os.path.join(global_config.data_save_dir, 'countywise_cum_cases')

config.countywise_cumulative_cases_file_format = \
    '%Y-%m-%d-cum-countywise-cases.csv'

config.get_countywise_cumulative_cases_file = \
    get_file_func(config.countywise_cumulative_cases_root,
                  config.countywise_cumulative_cases_file_format)

config.countywise_cumulative_cases_mean_file = \
    os.path.join(config.countywise_cumulative_cases_root, 'cum-countywise-cases-mean.pickle')

config.countywise_cumulative_cases_std_file = \
    os.path.join(config.countywise_cumulative_cases_root, 'cum-countywise-cases-std.pickle')

# sg mobility data
config.sg_mobility_root = \
    os.path.join(global_config.data_save_dir, 'sg_mobility')

config.sg_mobility_file_format = \
    '%Y-%m-%d-mobility.csv'

config.get_sg_mobility_file = \
    get_file_func(config.sg_mobility_root, config.sg_mobility_file_format)

config.sg_mobility_mean_file = \
    os.path.join(config.sg_mobility_root, 'mobility-mean.pickle')

config.sg_mobility_std_file = \
    os.path.join(config.sg_mobility_root, 'mobility-std.pickle')


sys.modules[__name__] = config
