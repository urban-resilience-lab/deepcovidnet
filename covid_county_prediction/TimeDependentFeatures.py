from covid_county_prediction.RawFeatures import RawFeatures
from datetime import date, timedelta
from math import ceil


class TimeDependentFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str, start_date: date, interval: timedelta):
        super(TimeDependentFeatures, self).__init__(raw_features, feature_name)
        self.start_date = start_date
        self.interval   = interval

    def extract_torch_tensor(self, county_fips: str, start_date: date, end_date: date):
        start_idx   = int((start_date - self.start_date) / self.interval)
        end_idx     = ceil((end_date - self.start_date) / self.interval)

        tensor = torch.zeros(end_idx - start_idx, self.raw_features.shape[1])

        for i in range(start_idx, end_idx):
            if self.raw_features[i].index.contains(county_fips):
                tensor[i] = self.raw_features[i].loc[county_fips].to_numpy()

        return tensor
