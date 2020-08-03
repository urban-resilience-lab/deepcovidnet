import torch.nn as nn
from deepcovidnet.EmbeddingModule import EmbeddingModule
from deepcovidnet.DeepFM import DeepFM


class CovidModule(nn.Module):
    def __init__(self, output_neurons):
        super(CovidModule, self).__init__()
        self.output_neurons     = output_neurons
        self.embedding_module   = EmbeddingModule()
        self.deep_fm            = DeepFM(output_neurons=self.output_neurons)

    def forward(self, features_dict):
        embeddings = self.embedding_module(features_dict)
        out = self.deep_fm(embeddings)
        return out
