from covid_county_prediction.TimeDependentFeatures import TimeDependentFeatures
from datetime import date, timedelta
import covid_county_prediction.config.CountyWiseTimeDependentFeaturesConfig as\
    config
import numpy as np
import torch
import logging


class CountyWiseTimeDependentFeatures(TimeDependentFeatures):
    def __init__(
        self, raw_features, feature_name: str, start_date: date,
        interval: timedelta, cur_type: str
    ):
        assert cur_type in config.types
        super(CountyWiseTimeDependentFeatures, self).__init__(
            raw_features, feature_name, start_date, interval
        )
        self.type = cur_type
        self.combined_features = [self]
        self.combined_start_date = self.start_date

    def extract_torch_tensor(
        self, county_fips: str, start_date: date, end_date: date
    ):
        assert start_date >= self.combined_start_date

        cur_date = start_date
        common_dates = []
        while cur_date < end_date:
            found_all_features = True
            for feature_index in range(len(self.combined_features)):
                if self.combined_features[feature_index].get_index(cur_date) is None:
                    found_all_features = False
                    break

            if found_all_features:
                common_dates.append(cur_date)

            cur_date += timedelta(1)

        assert len(common_dates), 'Features not combinable due to no common dates'

        logging.info(f'Found {len(common_dates)} time steps for combined countywise features')

        # init tensor of shape (num_time_steps, num_counties, num_features)
        tensor = torch.zeros(
            len(common_dates),
            self.raw_features[0].shape[0],
            len(self.combined_features)
        )

        for i, common_date in enumerate(common_dates):
            for feature_index in range(len(self.combined_features)):
                date_index = \
                    self.combined_features[feature_index].get_index(common_date)

                df = self.combined_features[feature_index].raw_features[date_index].fillna(0)
                cur_type = self.combined_features[feature_index].type

                if cur_type == config.cross_type:
                    features = df.loc[county_fips].to_numpy()
                elif cur_type == config.const_type:
                    features = np.squeeze(df.to_numpy(), axis=1)

                tensor[i, :, feature_index] = torch.tensor(features)

        return tensor

    def combine(self, other):
        self.combined_start_date = \
            min(self.combined_start_date, other.start_date)

        self.combined_features.append(other)
