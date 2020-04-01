from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import pandas as pd
import covid_county_prediction.constants as constants
import numpy as np

class BaseCountyDataset(Dataset, ABC):
    def __init__(self):
        # load all required data here
        self.poi_to_cbg = self.make_poi_to_cbg_dict()

    def make_poi_to_cbg_dict(self):
        df = pd.read_csv(constants.PLACE_COUNTY_CBG_FILE, usecols=['safegraph_place_id', 'countyFIPS'])
        df = df.dropna().astype({'countyFIPS':np.int32}).set_index('safegraph_place_id')

        return df['countyFIPS'].to_dict()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()
