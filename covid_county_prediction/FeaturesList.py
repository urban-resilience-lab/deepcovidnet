from covid_county_prediction.CountyWiseTimeDependentFeatures import CountyWiseTimeDependentFeatures
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

    def extract_torch_tensor(self, county_fips: str, start_date: date, end_date: date):
        tensors = []
        for i in range(len(self.features)):
            tensors.append(self.features[i].extract_torch_tensor(county_fips, start_date, end_date))

        return tensors
