import torch.nn as nn
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import torch
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config


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
                        layer = nn.Linear(features_dict[k].shape[i], hyperparams.embedding_size)
                        if torch.cuda.is_available():
                            layer = layer.cuda()

                        net.append(layer)
                    else:
                        layer = nn.Linear(features_dict[k].shape[i], 1)
                        if torch.cuda.is_available():
                            layer = layer.cuda()
                        net.append(layer)
                        net.append(nn.ReLU())
                        net.append(SqueezeLastDim())

                self.embedding_modules[k] = nn.Sequential(*net)

            self.are_layers_set = True

        for k in features_dict:
            out[k] = self.embedding_modules[k](features_dict[k])

        return out
