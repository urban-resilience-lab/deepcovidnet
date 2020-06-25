import torch
from covid_county_prediction.CovidModule import CovidModule
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
import torch.nn as nn


class CoralClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(CoralClassifier, self).__init__()
        self.fc = nn.Linear(in_features, 1)
        self.bias = nn.Parameter(torch.rand(out_features))

    def forward(self, processed_ftrs):
        return self.fc(processed_ftrs) + self.bias


class CovidCoralModule(CovidModule):
    # implementation of https://arxiv.org/pdf/1901.07884.pdf
    def __init__(self):
        super(CovidCoralModule, self).__init__(dataset_config.num_classifiers)

        self.deep_fm.classifier = CoralClassifier(
                                    self.deep_fm.classifier[0].in_features,
                                    self.output_neurons
                                )
