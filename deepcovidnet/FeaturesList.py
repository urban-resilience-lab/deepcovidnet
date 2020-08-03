from deepcovidnet.CountyWiseTimeDependentFeatures import CountyWiseTimeDependentFeatures
from datetime import date


class FeaturesList():
    def __init__(self, features):
        self.features = features

        for i in range(len(self.features)):
            if type(self.features[i]) == CountyWiseTimeDependentFeatures:
                indices_to_remove = []
                for j in range(i + 1, len(self.features)):
                    if type(self.features[j]) == CountyWiseTimeDependentFeatures:
                        self.features[i].combine(self.features[j])
                        indices_to_remove.append(j)

                for idx in indices_to_remove:
                    del self.features[idx]

                break

        self.key_to_feature = {}
        for i in range(len(self.features)):
            self.key_to_feature[self.get_key(self.features[i], i)] = self.features[i]

    def get_key(self, feature, idx):
        return \
            self.features[idx].feature_name + f'_{str(idx).zfill(2)}_' + \
            self.features[idx].__class__.__name__

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def extract_torch_tensors(self, county_fips: str, start_date: date, end_date: date):
        tensors = {}
        for i in range(len(self.features)):
            tensors[self.get_key(self.features[i], i)] = \
                self.features[i].extract_torch_tensor(
                    county_fips, start_date, end_date
                ).float()

        return tensors
