from deepcovidnet.TimeDependentFeatures import TimeDependentFeatures
from datetime import date, timedelta
import deepcovidnet.config.CountyWiseTimeDependentFeaturesConfig as\
    config
import deepcovidnet.config.features_config as features_config
import numpy as np
import torch


class CountyWiseTimeDependentFeatures(TimeDependentFeatures):
    def __init__(
        self, raw_features, feature_name: str, start_date: date,
        interval: timedelta, cur_type: str, feature_saver
    ):
        assert cur_type in config.types
        super(CountyWiseTimeDependentFeatures, self).__init__(
            raw_features, feature_name, start_date, interval, feature_saver
        )
        self.type = cur_type
        self.combined_features = [self]
        self.combined_start_date = self.start_date
        self.max_interval = self.interval

    def extract_torch_tensor(
        self, county_fips: str, start_date: date, end_date: date
    ):
        assert start_date >= self.combined_start_date

        cur_date = start_date
        common_dates = []
        while (end_date - cur_date).days >= self.max_interval.days:
            found_all_features = True
            for feature_index in range(len(self.combined_features)):
                if self.combined_features[feature_index].get_index(cur_date) is None:
                    found_all_features = False
                    break

            if found_all_features:
                common_dates.append(cur_date)

            cur_date += timedelta(1)

        assert len(common_dates), 'Features not combinable due to no common dates'

        # init tensor of shape (num_counties, num_time_steps, num_features)
        tensor = torch.zeros(
            self.raw_features[0].shape[0],
            len(common_dates),
            len(self.combined_features)
        )

        for i, common_date in enumerate(common_dates):
            for feature_index in range(len(self.combined_features)):
                date_index = \
                    self.combined_features[feature_index].get_index(common_date)

                df = self.combined_features[feature_index].raw_features[date_index]

                cur_type = self.combined_features[feature_index].type

                if cur_type == config.cross_type:
                    assert (df.columns == features_config.county_info.index).all()
                    features = \
                        df.values[features_config.county_to_iloc[county_fips]]
                elif cur_type == config.const_type:
                    assert df.shape[1] == 1
                    features = np.squeeze(df.to_numpy(), axis=1)

                tensor[:, i, feature_index] = torch.tensor(features)

        return tensor

    def combine(self, other):
        self.combined_start_date = \
            min(self.combined_start_date, other.start_date)

        self.max_interval = \
            max(self.max_interval, other.interval)

        self.combined_features.append(other)

    def get_feature_name(self, idx):
        return self.combined_features[idx].feature_name
