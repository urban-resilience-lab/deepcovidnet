from RawFeatures import RawFeatures

class TimeDependentFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str):
        super(TimeDependentFeatures, self).__init__(raw_features, feature_name)

    def extract(self):
        pass