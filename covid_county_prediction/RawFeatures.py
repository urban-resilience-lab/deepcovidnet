import covid_county_prediction.config.RawFeaturesConfig as config
import covid_county_prediction.config.features_config as features_config
import pandas as pd

class RawFeatures:
    def __init__(self, raw_features, feature_name: str, feature_type: config.feature_type):
        '''
        Args:
            raw_features: a list of Dataframes for time dependent features
                        or just a Dataframe for constant features 
        '''

        self.feature_type = feature_type
        self.feature_name = feature_name
        self.features = self.process_features(raw_features)

    def process_features(self, raw_features):
        return self.get_features_with_index(features_config.county_info.index, raw_features)

    def get_features_with_index(self, index, raw_features):
        index_df = pd.DataFrame(index=index)

        if type(raw_features) == type([]):
            ans = []
            for df in raw_features:
                ans.append(
                    index_df.merge(df, how='left', left_index=True, right_index=True, suffixes=('', ''))
                )

            if len(ans) > 1:
                return ans
            else:
                self.feature_type = config.feature_type.CONSTANTS
                return ans[0]

        # raw_features is a df
        return index_df.merge(raw_features, how='left', left_index=True, right_index=True, suffixes=('', ''))

    def keep_features_with_labels(self, labels_df):
        return self.get_features_with_index(labels_df.index, self.raw_features)