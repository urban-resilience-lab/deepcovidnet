from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config
import datetime
import os


config = Config('Config for DataSaver')


class FeatureSaver:
    def __init__(self, root, keyword, time_dependence=True):
        self.root = os.path.join(global_config.data_save_dir, root)
        self.keyword = keyword
        self.time_dependence = time_dependence

        if self.time_dependence:
            self.file_format = f'%Y-%m-%d-{self.keyword}.csv'
            self.mean_path = os.path.join(self.root,
                                          f'{self.keyword}-mean.pickle')
            self.std_path = os.path.join(self.root,
                                         f'{self.keyword}-std.pickle')
            self.save_file = None
        else:
            self.file_format = f'{self.keyword}.csv'
            self.mean_path = None
            self.std_path = None
            self.save_file = os.path.join(self.root, f'{self.keyword}.csv')

    def get_file_func(self):
        return lambda d: d.strftime(
            os.path.join(self.root, self.file_format)
        )


# census data
config.census_data = FeatureSaver(
                        root='sg_census_data',
                        keyword='sg-census-data',
                        time_dependence=False
                    )

# monthly data
config.sg_patterns_monthly = FeatureSaver(
                                root='monthly_patterns',
                                keyword='monthly-patterns'
                            )

# social distancing data
config.sg_social_distancing = FeatureSaver(
                                root='social_distancing',
                                keyword='social-distancing'
                            )

# weather data
config.weather = FeatureSaver(root='weather_data', keyword='weather')

# num cases data
config.num_cases = FeatureSaver(root='num_cases', keyword='num-cases')

# dilation index
config.dilation_index = FeatureSaver(root='dilation_index', keyword='di')

# countywise cumulative cases data
config.countywise_cumulative_cases = FeatureSaver(
                                        root='countywise_cum_cases',
                                        keyword='cum-countywise-cases'
                                    )

# sg mobility data
config.sg_mobility = FeatureSaver(root='sg_mobility', keyword='mobility')


sys.modules[__name__] = config
