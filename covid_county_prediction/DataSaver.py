from covid_county_prediction.RawFeatureExtractor import RawFeatureExtractor
import covid_county_prediction.config.global_config as global_config
import os
import covid_county_prediction.config.DataSaverConfig as config
import logging


class DataSaver(RawFeatureExtractor):
    def __init__(self):
        if not os.path.exists(global_config.data_save_dir):
            os.mkdir(global_config.data_save_dir)
        super(DataSaver, self).__init__()

    def save_census_data(self, overwrite=False):
        if not os.path.exists(config.census_data.root):
            os.mkdir(config.census_data.root)

        df = self.read_census_data().raw_features
        self._save_df(config.census_data.save_file, df, overwrite)

    def save_sg_patterns_monthly(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_sg_patterns_monthly,
            config.sg_patterns_monthly,
            overwrite
        )

    def save_sg_social_distancing(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_sg_social_distancing,
            config.sg_social_distancing,
            overwrite
        )

    def save_weather_data(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_weather_data,
            config.weather,
            overwrite
        )

    def save_num_cases(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_num_cases,
            config.num_cases,
            overwrite
        )

    def save_dilation_index(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_dilation_index,
            config.dilation_index,
            overwrite
        )

    def save_countywise_cumulative_cases(self, start_date, end_date,
                                         overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_countywise_cumulative_cases,
            config.countywise_cumulative_cases,
            overwrite
        )

    def save_sg_mobility_incoming(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_sg_mobility_incoming,
            config.sg_mobility,
            overwrite
        )

    def _save_time_dep_features(self, start_date, end_date, get_features,
                                saver_config, overwrite):
        if not os.path.exists(saver_config.root):
            os.mkdir(saver_config.root)

        features = get_features(start_date, end_date)

        for i in range(len(features.raw_features)):
            save_file = saver_config.get_file_func()(features.get_date(i))
            self._save_df(save_file, features.raw_features[i], overwrite)

    def _save_df(self, save_file, df, overwrite=False):
        if os.path.exists(save_file):
            logging.warning(f'{save_file} already exists!')
            if not overwrite:
                return

        df.to_csv(save_file, index_label='fips')
        logging.info(f'Saved {save_file}!')
