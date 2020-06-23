from covid_county_prediction.RawFeatures import RawFeatures
from datetime import date, timedelta
import torch
import pickle
import pandas as pd
import os
import covid_county_prediction.config.features_config as features_config


class TimeDependentFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str, start_date: date,
                 interval: timedelta, feature_saver):
        super(TimeDependentFeatures, self).__init__(raw_features, feature_name, feature_saver)
        self.start_date = start_date
        self.interval   = interval

    def get_date(self, i):
        assert i >= 0 and i < len(self.raw_features)
        return self.start_date + i * self.interval

    def get_index(self, cur_date):
        does_index_exist = \
            (cur_date - self.start_date).days % self.interval.days == 0

        end_date = self.get_date(len(self.raw_features) - 1) + self.interval

        if cur_date < self.start_date or (not does_index_exist) or\
           (end_date - cur_date).days < self.interval.days:
            return None

        return int((cur_date - self.start_date) / self.interval)

    def extract_torch_tensor(self, county_fips: str, start_date: date,
                             end_date: date):
        start_idx = self.get_index(start_date)
        end_idx   = self.get_index(end_date)

        tensor = torch.zeros(end_idx - start_idx, self.raw_features[0].shape[1])

        for i in range(start_idx, end_idx):
            if county_fips in self.raw_features[i].index:
                np_tensor = \
                    self.raw_features[i].values[
                        features_config.county_to_iloc[county_fips]
                    ]

                tensor[i - start_idx] = torch.tensor(np_tensor)

        return tensor

    def normalize(self, mean=None, std=None):
        if (mean is None) or (std is None):
            concatenated = pd.concat(self.raw_features)
        if mean is None:
            mean = concatenated.mean()
        if std is None:
            std = concatenated.std()

        for i in range(len(self.raw_features)):
            self.raw_features[i] = ((self.raw_features[i] - mean) / std).fillna(0)

        return mean, std

    def get_feature_name(self, idx):
        return self.feature_name + '__' + self.raw_features[0].columns[idx]
