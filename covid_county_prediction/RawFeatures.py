import covid_county_prediction.config.RawFeaturesConfig as config
import covid_county_prediction.config.features_config as features_config
import pandas as pd
from abc import ABC, abstractmethod
from datetime import date


class RawFeatures(ABC):
    def __init__(self, raw_features, feature_name: str):
        '''
        Args:
            raw_features: a list of Dataframes or just a Dataframe
        '''
        self.feature_name = feature_name
        self.raw_features = self.process_features(raw_features)

    def process_features(self, raw_features):
        return self.get_features_with_index(features_config.county_info.index,
                                            raw_features)

    def get_features_with_index(self, index, raw_features):
        index_df = pd.DataFrame(index=index)

        if isinstance(raw_features, list):
            ans = []
            for df in raw_features:
                ans.append(
                    index_df.merge(
                        df, how='left', left_index=True,
                        right_index=True, suffixes=('', '')
                    )
                )

            return ans
        # raw_features is a df
        return index_df.merge(
            raw_features, how='left', left_index=True,
            right_index=True, suffixes=('', '')
        )

    def keep_features_with_labels(self, labels_df):
        return self.get_features_with_index(labels_df.index, self.raw_features)

    @abstractmethod
    def extract_torch_tensor(self, county_fips: str, start_date: date,
                             end_date: date):
        raise NotImplementedError()
