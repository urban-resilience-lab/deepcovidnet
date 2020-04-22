import torch.nn as nn
from covid_county_prediction.EmbeddingModule import EmbeddingModule
from covid_county_prediction.DeepFM import DeepFM


class CovidModule():
    def __init__(self):
        super(CovidModule, self).__init__()
        self.embedding_module   = EmbeddingModule()
        self.deep_fm            = DeepFM()

    def forward(self, features_dict):
        return self.deep_fm(self.embedding_module(features_dict))
