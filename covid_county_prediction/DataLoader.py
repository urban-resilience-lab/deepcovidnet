from covid_county_prediction.DataSaver import DataSaver
import covid_county_prediction.config.DataSaverConfig as saver_config
import covid_county_prediction.config.RawFeatureExtractorConfig as rfe_config
import os
import pandas as pd
from covid_county_prediction.ConstantFeatures import ConstantFeatures
from covid_county_prediction.CountyWiseTimeDependentFeatures import CountyWiseTimeDependentFeatures
from covid_county_prediction.TimeDependentFeatures import TimeDependentFeatures
from datetime import timedelta
import time
import logging
from covid_county_prediction.utils import timed_logger_decorator


class DataLoader(DataSaver):
    def __init__(self):
        super(DataLoader, self).__init__()

    @timed_logger_decorator
    def load_census_data(self):
        self._save_if_not_saved(
            saver_config.census_data.save_file,
            self.save_census_data
        )

        df = pd.read_csv(
            saver_config.census_data.save_file, dtype={'fips': str}
        ).set_index('fips')

        return ConstantFeatures(df, 'open_census_data',
                                feature_saver=saver_config.census_data)

    @timed_logger_decorator
    def load_sg_patterns_monthly(self, start_date, end_date):
        return self._load_time_dep_features(
            start_date, end_date, saver_config.sg_patterns_monthly.get_file_func(),
            self.save_sg_patterns_monthly, TimeDependentFeatures,
            'monthly_patterns',
            feature_saver=saver_config.sg_patterns_monthly
        )

    @timed_logger_decorator
    def load_sg_social_distancing(self, start_date, end_date):
        return self._load_time_dep_features(
            start_date, end_date, saver_config.sg_social_distancing.get_file_func(),
            self.save_sg_social_distancing, TimeDependentFeatures,
            'social_distancing',
            feature_saver=saver_config.sg_social_distancing
        )

    @timed_logger_decorator
    def load_weather_data(self, start_date, end_date):
        return self._load_time_dep_features(
            start_date, end_date, saver_config.weather.get_file_func(),
            self.save_weather_data, TimeDependentFeatures,
            'weather_data',
            feature_saver=saver_config.weather
        )

    @timed_logger_decorator
    def load_num_cases(self, start_date, end_date):
        return self._load_time_dep_features(
            start_date, end_date, saver_config.num_cases.get_file_func(),
            self.save_num_cases, TimeDependentFeatures,
            'num_cases',
            feature_saver=saver_config.num_cases
        )

    @timed_logger_decorator
    def load_dilation_index(self, start_date, end_date):
        return self._load_time_dep_features(
            start_date, end_date, saver_config.dilation_index.get_file_func(),
            self.save_dilation_index, TimeDependentFeatures, 'di',
            feature_saver=saver_config.dilation_index
        )

    @timed_logger_decorator
    def load_countywise_cumulative_cases(self, start_date, end_date):
        return self._load_time_dep_features(
            start_date, end_date,
            saver_config.countywise_cumulative_cases.get_file_func(),
            self.save_countywise_cumulative_cases,
            CountyWiseTimeDependentFeatures,
            'countywise_cumulative_cases',
            cur_type='CONSTANT',
            feature_saver=saver_config.countywise_cumulative_cases
        )

    @timed_logger_decorator
    def load_sg_mobility_incoming(self, start_date, end_date):
        d = rfe_config.sg_patterns_weekly_reader.get_file_date(start_date)
        interval = timedelta(7)

        if d < start_date:
            d += interval

        return self._load_time_dep_features(
            d,
            end_date,
            saver_config.sg_mobility.get_file_func(),
            self.save_sg_mobility_incoming,
            CountyWiseTimeDependentFeatures,
            'countywise_mobility',
            cur_type='CROSS',
            interval=interval,
            feature_saver=saver_config.sg_mobility
        )

    def _load_time_dep_features(self, start_date, end_date, get_path, saver,
                                feature_type, feature_name, feature_saver,
                                cur_type=None, interval=timedelta(1)):
        self._save_if_not_saved(get_path, saver, start_date, end_date, interval)

        dfs = []
        cur_date = start_date
        while((end_date - cur_date).days >= interval.days):
            dfs.append(
                pd.read_csv(
                    get_path(cur_date), dtype={'fips': str}
                ).set_index('fips')
            )
            cur_date += interval

        if feature_type == TimeDependentFeatures:
            return feature_type(dfs, feature_name, start_date, interval,
                                feature_saver=feature_saver)
        else:
            return feature_type(dfs, feature_name, start_date, interval,
                                cur_type=cur_type, feature_saver=feature_saver)

    def _save_if_not_saved(self, saved_path_or_get_path, saver,
                           start_date=None, end_date=None,
                           interval=timedelta(1)):
        if isinstance(saved_path_or_get_path, str):
            if not os.path.exists(saved_path_or_get_path):
                if start_date is None and end_date is None:
                    saver()
                elif start_date is not None and end_date is not None:
                    saver(start_date, end_date)
                else:
                    raise Exception('either both start_date, end_date must be'
                                    'provided or none must be')
        else:  # saved_path_or_get_path is a function
            cur_date = start_date
            while cur_date < end_date:
                if not os.path.exists(saved_path_or_get_path(cur_date)):
                    saver(cur_date, end_date)
                cur_date += interval
