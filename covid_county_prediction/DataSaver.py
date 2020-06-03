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
        if not os.path.exists(config.census_data_root):
            os.mkdir(config.census_data_root)

        df = self.read_census_data().raw_features
        self._save_df(config.census_data_path, df, overwrite)

    def save_sg_patterns_monthly(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_sg_patterns_monthly,
            config.sg_patterns_monthly_root,
            config.get_sg_patterns_monthly_file,
            overwrite
        )

    def save_sg_social_distancing(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_sg_social_distancing,
            config.sg_social_distancing_root,
            config.get_sg_social_distancing_file,
            overwrite
        )

    def save_weather_data(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_weather_data,
            config.weather_root,
            config.get_weather_file,
            overwrite
        )

    def read_sg_mobility_incoming(self, start_date, end_date, overwrite=False):
        self._save_time_dep_features(
            start_date,
            end_date,
            self.read_sg_mobility_incoming,
            config.sg_mobility_root,
            config.get_sg_mobility_file,
            overwrite
        )

    def _save_time_dep_features(self, start_date, end_date, get_features,
                                save_root, get_save_file, overwrite):
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        features = get_features(start_date, end_date)

        for i in range(len(features.raw_features)):
            save_file = get_save_file(features.get_date(i))
            self._save_df(save_file, features.raw_features[i], overwrite)

    def _save_df(self, save_file, df, overwrite=False):
        if os.path.exists(save_file):
            logging.warning(f'{save_file} already exists!')
            if not overwrite:
                return

        df.to_csv(save_file, index_label='fips')
        logging.info(f'Saved {save_file}!')
