from RawFeatures import RawFeatures

class ConstantFeatures(RawFeatures):
    def __init__(self, raw_features, feature_name: str):
        super(ConstantFeatures, self).__init__(raw_features, feature_name)

    def extract(self):
        pass