from covid_county_prediction.RawFeatures import RawFeatures
import torch
from datetime import date


class ConstantFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str):
        super(ConstantFeatures, self).__init__(raw_features, feature_name)

    def extract_torch_tensor(self, county_fips: str, start_date: date, end_date: date):
        if self.raw_features.index.contains(county_fips):
            return torch.tensor(self.raw_features.loc[county_fips].to_numpy())

        return torch.zeros(self.raw_features.shape[1])
