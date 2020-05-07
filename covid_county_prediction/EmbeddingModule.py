import torch.nn as nn
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import torch
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
from covid_county_prediction.ConstantFeatures import ConstantFeatures
from covid_county_prediction.TimeDependentFeatures import TimeDependentFeatures
from covid_county_prediction.CountyWiseTimeDependentFeatures import CountyWiseTimeDependentFeatures

# TODO: Parameters must be initialized at init time


class EmbeddingModule(nn.Module):
    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.are_layers_set = False
        self.embedding_modules = nn.ModuleDict()

    def forward(self, features_dict):
        assert dataset_config.labels_key not in features_dict

        out = {}

        if not self.are_layers_set:
            class SqueezeLastDim(nn.Module):
                def forward(self, x):
                    return x.squeeze(-1)

            for k in features_dict:
                net = []
                for i in range(features_dict[k].dim() - 1, 0, -1):
                    if i == 1:
                        net.append(nn.Linear(features_dict[k].shape[i], hyperparams.embedding_size))
                    else:
                        net.append(nn.Linear(features_dict[k].shape[i], 1))
                        net.append(nn.ReLU())
                        net.append(SqueezeLastDim())

                self.embedding_modules[k] = nn.Sequential(*net)

            self.are_layers_set = True

        for k in features_dict:
            out[k] = self.embedding_modules[k](features_dict[k])

        return out
