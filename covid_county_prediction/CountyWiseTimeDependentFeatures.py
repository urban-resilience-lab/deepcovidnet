from covid_county_prediction.RawFeatures import RawFeatures
from datetime import date


class CountyWiseTimeDependentFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str):
        super(CountyWiseTimeDependentFeatures, self).__init__(raw_features, feature_name)

    def extract_torch_tensor(self, county_fips: str, start_date: date, end_date: date):
        pass

    def combine(self, other):
        pass
