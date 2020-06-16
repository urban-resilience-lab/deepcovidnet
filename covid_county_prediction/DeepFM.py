import torch.nn as nn
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
import torch


class DeepFM(nn.Module):
    def __init__(self, output_neurons,
                 feature_dim=hyperparams.embedding_size,
                 num_features=dataset_config.num_features):
        super(DeepFM, self).__init__()

        assert num_features > 1

        self.output_neurons = output_neurons
        self.feature_dim = feature_dim
        self.num_features = num_features

        self.deep_processor = nn.Sequential(
                nn.Linear(self.num_features * self.feature_dim, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, hyperparams.higher_order_features_size)
            )

        self.classifier = nn.Sequential(
            nn.Linear(
                hyperparams.higher_order_features_size +
                int(self.num_features * (self.num_features - 1) / 2),
                self.output_neurons
            )
        )

        self.second_order_interactions = None

    def forward(self, features_dict):
        '''
        Args:
            feature_list: dict of PyTorch tensors of shape (batch_size, self.feature_dim)
        '''
        assert dataset_config.labels_key not in features_dict

        sorted_keys = sorted(features_dict)

        # FM Part
        self.second_order_interactions = torch.empty(
                                        features_dict[list(features_dict.keys())[0]].shape[0],
                                        int(self.num_features * (self.num_features - 1) / 2),
                                        requires_grad=False
                                    )

        idx = 0
        for i in range(len(sorted_keys)):
            for j in range(i + 1, len(sorted_keys)):
                self.second_order_interactions[:, idx] = \
                    torch.bmm(
                        features_dict[sorted_keys[i]].unsqueeze(1),
                        features_dict[sorted_keys[j]].unsqueeze(2)
                    ).squeeze()
                idx += 1

        if torch.cuda.is_available():
            self.second_order_interactions = self.second_order_interactions.cuda()

        # Deep Part
        concatenated_features = [features_dict[sorted_keys[i]] for i in range(len(sorted_keys))]
        concatenated_features = torch.cat(concatenated_features, dim=1)
        higher_order_interactions = self.deep_processor(concatenated_features)

        interactions = torch.cat([higher_order_interactions, self.second_order_interactions], dim=1)
        return self.classifier(interactions)
