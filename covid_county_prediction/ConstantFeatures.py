from covid_county_prediction.RawFeatures import RawFeatures
import torch
from datetime import date
import covid_county_prediction.config.features_config as features_config


class ConstantFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str, feature_saver):
        super(ConstantFeatures, self).__init__(raw_features, feature_name,
                                               feature_saver)

    def extract_torch_tensor(self, county_fips: str, start_date: date,
                             end_date: date):
        if county_fips in self.raw_features.index:
            return torch.tensor(
                        self.raw_features.values[
                            features_config.county_to_iloc[county_fips]
                        ]
                   )

        return torch.zeros(self.raw_features.shape[1])

    def normalize(self, mean=None, std=None, fill_na=True):
        if mean is None:
            mean = self.raw_features.mean()
        if std is None:
            std = self.raw_features.std()

        self.raw_features = (self.raw_features - mean) / std

        if fill_na:
            self.raw_features = self.raw_features.fillna(0)

        return mean, std

    def get_feature_name(self, idx):
        return self.feature_name + '__' + self.raw_features.columns[idx]
