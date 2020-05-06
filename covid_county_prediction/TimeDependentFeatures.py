from covid_county_prediction.RawFeatures import RawFeatures
from datetime import date, timedelta
from math import ceil
import torch


class TimeDependentFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str, start_date: date, interval: timedelta):
        super(TimeDependentFeatures, self).__init__(raw_features, feature_name)
        self.start_date = start_date
        self.interval   = interval

    def get_date(self, i):
        return self.start_date + i * self.interval

    def extract_torch_tensor(self, county_fips: str, start_date: date, end_date: date):
        start_idx   = int((start_date - self.start_date) / self.interval)
        end_idx     = ceil((end_date - self.start_date) / self.interval)

        tensor = torch.zeros(end_idx - start_idx, self.raw_features[0].shape[1])

        for i in range(start_idx, end_idx):
            if county_fips in self.raw_features[i].index:
                tensor[i - start_idx] = \
                    torch.tensor(self.raw_features[i].loc[county_fips].fillna(0).to_numpy())

        return tensor
