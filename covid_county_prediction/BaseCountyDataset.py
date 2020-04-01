from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import glob
import pandas as pd
import covid_county_prediction.constants as constants
import numpy as np


class BaseCountyDataset(Dataset, ABC):
    def __init__(self):
        # load all required data here
        self.poi_to_cbg = self.make_poi_to_cbg_dict()

    def make_poi_to_cbg_dict(self):
        df = pd.read_csv(constants.PLACE_COUNTY_CBG_FILE, usecols=['safegraph_place_id', 'countyFIPS'])
        df = df.dropna().astype({'countyFIPS': np.int32}).set_index('safegraph_place_id')

        return df['countyFIPS'].to_dict()

    @staticmethod
    def make_census_dict():
        if __name__ == '__main__':
            main_df = pd.DataFrame()
            for file in glob.glob(constants.PATH_TO_SAFEGRAPH_OPEN_CENSUS_DATA + "cbg_*.csv"):
                df = pd.read_csv(file, converters={
                    'census_block_group': lambda x: str(x)})  # The converter is used to retain leading zeros

                # Convert census block groups to FIPS codes and use as index named FIPS
                df.index = df['census_block_group'].apply(lambda x: x[:5]).astype(
                    int)  # Save only first five chars of index
                df.drop('census_block_group', axis=1, inplace=True)
                df.index.name = "FIPS"

                # Aggregate data by FIPS codes
                df = df.groupby("FIPS").sum()
                main_df = main_df.join(df, how="outer")
            return main_df.to_dict()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()
