import torch.nn as nn
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
import torch


class DeepFM(nn.Module):
    def __init__(self, num_classes, feature_dim=hyperparams.embedding_size):
        super(DeepFM, self).__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.are_layers_set = False

    def forward(self, features_dict):
        '''
        Args:
            feature_list: dict of PyTorch tensors of shape (batch_size, self.feature_dim) 
        '''
        assert dataset_config.labels_key not in features_dict

        if not self.are_layers_set:
            self.num_features = len(features_dict)
            self.deep_processor = nn.Sequential(
                nn.Linear(self.num_features * self.feature_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, hyperparams.higher_order_features_size)
            )
            self.classifier = nn.Sequential(
                nn.Linear(
                    hyperparams.higher_order_features_size +
                    self.num_features * (self.num_features - 1) / 2,
                    hyperparams.num_classes
                )
            )

            self.are_layers_set = True

        sorted_keys = sorted(features_dict)

        # FM Part
        second_order_interactions = torch.zeros(
                                        features_dict[list(features_dict.keys())[0]].shape[0],
                                        self.num_features * (self.num_features - 1) / 2, 
                                        requires_grad=True
                                    )

        idx = 0
        for i in range(len(sorted_keys)):
            for j in range(i + 1, len(sorted_keys)):
                second_order_interactions[:, idx] = \
                    torch.bmm(
                        features_dict[sorted_keys[i]].unsqueeze(1),
                        features_dict[sorted_keys[j]].unsqueeze(2)
                    ).squeeze()
                idx += 1

        # Deep Part
        concatenated_features = [features_dict[sorted_keys[i]] for i in range(len(sorted_keys))]
        concatenated_features = torch.cat(concatenated_features)
        higher_order_interactions = self.deep_processor(concatenated_features)

        interactions = torch.cat([higher_order_interactions, second_order_interactions])
        return self.classifier(interactions)
