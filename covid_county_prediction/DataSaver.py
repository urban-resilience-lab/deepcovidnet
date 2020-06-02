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
        if not os.path.exists(config.sg_patterns_monthly_root):
            os.mkdir(config.sg_patterns_monthly_root)

        features = self.read_sg_patterns_monthly(start_date, end_date)

        for i in range(len(features.raw_features)):
            save_file = config.get_sg_patterns_monthly_file(features.get_date(i))
            self._save_df(save_file, features.raw_features[i], overwrite)

    def save_sg_social_distancing(self, start_date, end_date, overwrite=False):
        if not os.path.exists(config.sg_social_distancing_root):
            os.mkdir(config.sg_social_distancing_root)

        features = self.read_sg_social_distancing(start_date, end_date)

        for i in range(len(features.raw_features)):
            save_file = config.get_sg_social_distancing_file(features.get_date(i))
            self._save_df(save_file, features.raw_features[i], overwrite)

    def _save_df(self, save_file, df, overwrite=False):
        if os.path.exists(save_file):
            logging.warning(f'{save_file} already exists!')
            if not overwrite:
                return

        df.to_csv(save_file, index_label='fips')
        logging.info(f'Saved {save_file}!')
