from covid_county_prediction.TimeDependentFeatures import TimeDependentFeatures
from datetime import date, timedelta
import covid_county_prediction.config.CountyWiseTimeDependentFeaturesConfig as\
    config
import numpy as np
from math import ceil
import torch


class CountyWiseTimeDependentFeatures(TimeDependentFeatures):
    def __init__(
        self, raw_features, feature_name: str, start_date: date,
        interval: timedelta, cur_type: str
    ):
        assert cur_type in config.types
        super(CountyWiseTimeDependentFeatures, self).__init__(
            raw_features, feature_name, start_date, interval
        )
        self.combined_features = [(self.raw_features, self.type)]
        self.type = cur_type

    def extract_torch_tensor(
        self, county_fips: str, start_date: date, end_date: date
    ):
        start_idx = self.get_index(start_date)
        end_idx   = self.get_index(end_date, round_fn=ceil)

        # init tensor of shape (num_type_steps, num_counties, num_features)
        tensor = torch.zeros(
            end_idx - start_idx,
            self.raw_features[0].shape[0],
            len(self.combined_features)
        )

        for date_index in range(start_idx, end_idx):
            for feature_index in range(len(self.combined_features)):
                df = self.combined_features[feature_index][0][date_index]\
                        .fillna(0)
                cur_type = self.combined_features[feature_index][1]

                if cur_type == config.cross_type:
                    features = df.loc[county_fips].to_numpy()
                elif cur_type == config.const_type:
                    features = np.squeeze(df.to_numpy(), axis=1)

                tensor[date_index - start_idx, :, feature_index] = features

        return tensor

    def combine(self, other):
        self.combined_features.append((other.raw_features, other.type))
