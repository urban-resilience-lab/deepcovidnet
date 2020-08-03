import torch.nn as nn
import deepcovidnet.config.model_hyperparam_config as hyperparams
import deepcovidnet.config.CovidCountyDatasetConfig as dataset_config
import torch


class BaseDeepProcessor(nn.Module):
    def __init__(self, in_features, out_features):
        super(BaseDeepProcessor, self).__init__()
        self.in_features = in_features
        self.out_features = out_features


class TunableDeepProcessor(BaseDeepProcessor):
    def __init__(self, in_features, out_features):
        super(TunableDeepProcessor, self).__init__(in_features, out_features)
        self.intermediate_size = hyperparams.deep_intermediate_size
        self.num_layers = hyperparams.deep_layers

        self.net = []

        for i in range(self.num_layers):
            in_f, out_f = self.intermediate_size, self.intermediate_size
            if i == 0:
                in_f = self.in_features
            if i == self.num_layers - 1:
                out_f = self.out_features

            self.net.append(nn.Linear(in_f, out_f))
            if i != self.num_layers - 1:
                self.net.append(nn.SELU())
                self.net.append(nn.AlphaDropout(hyperparams.alpha_dropout_prob))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class FixedDeepProcessor(BaseDeepProcessor):
    def __init__(self, in_features, out_features):
        super(FixedDeepProcessor, self).__init__(in_features, out_features)
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(hyperparams.dropout_prob),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        return self.net(x)


class DeepFM(nn.Module):
    def __init__(self, output_neurons):
        super(DeepFM, self).__init__()

        self.output_neurons = output_neurons
        self.feature_dim = hyperparams.embedding_size
        self.num_features = dataset_config.num_features

        self.deep_processor = TunableDeepProcessor(
                                self.num_features * self.feature_dim,
                                hyperparams.higher_order_features_size
                            )

        self.classifier = FixedDeepProcessor(
                            hyperparams.higher_order_features_size +
                            int(self.num_features * (self.num_features - 1) / 2) +
                            self.feature_dim,
                            self.output_neurons
                        )

        self.so_int_labels = None

        self.so_int = None

    def compute_soi(self, features_dict):
        sorted_keys = sorted(features_dict)

        # FM Part
        if self.so_int_labels is None:
            idx = 0
            self.so_int_labels = {}
            for i in range(len(sorted_keys)):
                for j in range(i + 1, len(sorted_keys)):
                    self.so_int_labels[idx] = \
                        [sorted_keys[i], sorted_keys[j]]
                    idx += 1

        self.so_int = torch.empty(
                        features_dict[list(features_dict.keys())[0]].shape[0],
                        int(self.num_features * (self.num_features - 1) / 2),
                        requires_grad=False
                    )

        idx = 0
        for i in range(len(sorted_keys)):
            for j in range(i + 1, len(sorted_keys)):
                self.so_int[:, idx] = \
                    torch.bmm(
                        features_dict[sorted_keys[i]].unsqueeze(1),
                        features_dict[sorted_keys[j]].unsqueeze(2)
                    ).squeeze()
                idx += 1

        if torch.cuda.is_available():
            self.so_int = self.so_int.cuda()

    def compute_deep(self, features_dict):
        sorted_keys = sorted(features_dict)

        concatenated_features = [features_dict[sorted_keys[i]] for i in range(len(sorted_keys))]
        concatenated_features = torch.cat(concatenated_features, dim=1)
        higher_order_interactions = self.deep_processor(concatenated_features)

        classifier_in = [higher_order_interactions, self.so_int]
        classifier_in += \
            [torch.stack([features_dict[k] for k in sorted_keys]).sum(0)]
        classifier_in = torch.cat(classifier_in, dim=1)

        return self.classifier(classifier_in)

    def forward(self, features_dict):
        '''
        Args:
            feature_dict: dict of PyTorch tensors of shape (batch_size, self.feature_dim)
        '''
        assert dataset_config.labels_key not in features_dict

        # FM Part
        self.compute_soi(features_dict)

        # Deep Part
        return self.compute_deep(features_dict)
