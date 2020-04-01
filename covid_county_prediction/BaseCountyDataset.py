from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseCountyDataset(Dataset, ABC):
    def __init__(self):
        # load all required data here
        pass

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()