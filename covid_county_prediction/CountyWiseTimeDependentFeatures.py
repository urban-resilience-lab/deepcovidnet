from RawFeatures import RawFeatures


class CountyWiseTimeDependentFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str):
        super(CountyWiseTimeDependentFeatures, self).__init__(raw_features, feature_name)

    def extract_torch_tensor(self, county_fips: str, start_date: date, end_date: date):
        pass

    def combine(self, other):
        pass