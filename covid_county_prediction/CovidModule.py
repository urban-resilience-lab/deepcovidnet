import torch.nn as nn
from covid_county_prediction.EmbeddingModule import EmbeddingModule
from covid_county_prediction.DeepFM import DeepFM


class CovidModule(nn.Module):
    def __init__(self, output_neurons):
        super(CovidModule, self).__init__()
        self.embedding_module   = EmbeddingModule()
        self.deep_fm            = DeepFM(output_neurons=output_neurons)

    def forward(self, features_dict):
        embeddings = self.embedding_module(features_dict)
        out = self.deep_fm(embeddings)
        return out
