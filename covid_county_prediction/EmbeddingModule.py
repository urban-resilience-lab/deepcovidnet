import torch.nn as nn
import torch.nn.init as init
import torch
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
import math


class EmbeddingModule(nn.Module):
    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.are_layers_set = False
        self.embedding_modules = nn.ModuleDict()

    def forward(self, features_dict):
        assert dataset_config.labels_key not in features_dict

        out = {}

        if not self.are_layers_set:
            class ElementWiseProdCondense(nn.Module):
                def __init__(self, in_size):
                    super(ElementWiseProdCondense, self).__init__()
                    self.weight = nn.Parameter(torch.Tensor(in_size, hyperparams.embedding_size))
                    self.bias   = nn.Parameter(torch.Tensor(hyperparams.embedding_size))
                    self.reset_parameters()

                def reset_parameters(self):
                    # taken from https://pytorch.org/docs/master/_modules/torch/nn/modules/linear.html#Linear
                    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                    if self.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(self.bias, -bound, bound)

                def forward(self, x):
                    return torch.mul(x, self.weight).sum(axis=-2) + self.bias

            for k in features_dict:
                net = []
                is_first_done = False
                for i in range(features_dict[k].dim() - 1, 0, -1):
                    if not is_first_done:
                        layer = nn.Linear(
                                    features_dict[k].shape[i],
                                    hyperparams.embedding_size
                                )
                        is_first_done = True
                        net.append(layer)
                        net.append(nn.ReLU())
                    else:
                        net.append(ElementWiseProdCondense(features_dict[k].shape[i]))
                        net.append(nn.ReLU())

                if torch.cuda.is_available():
                    for i in range(len(net)):
                        net[i] = net[i].cuda()

                self.embedding_modules[k] = nn.Sequential(*net)

            self.are_layers_set = True

        for k in features_dict:
            out[k] = self.embedding_modules[k](features_dict[k])

        return out
