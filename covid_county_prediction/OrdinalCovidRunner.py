from covid_county_prediction.CovidModule import CovidModule
from covid_county_prediction.CovidRunner import CovidRunner
import torch
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import torch.nn as nn
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config


class OrdinalBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(OrdinalBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, labels):
        return self.bce_loss(
                pred.flatten().unsqueeze(1),
                labels.flatten().unsqueeze(1),
            )


class OrdinalCovidRunner(CovidRunner):
    def __init__(self, exp_name, load_path=None, sample_batch=None):
        net = CovidModule(output_neurons=dataset_config.num_classifiers)

        super(OrdinalCovidRunner, self).__init__(
            exp_name=exp_name,
            net=net,
            loss_fn=OrdinalBCEWithLogitsLoss(),
            load_path=load_path,
            sample_batch=sample_batch
        )

    def _get_extra_metrics(self, pred, labels):
        ordinal_labels = self.transform_labels(labels)
        classifier_acc = \
            (pred.sigmoid().round().flatten() == ordinal_labels.flatten()).sum().item() / pred.numel()

        metrics = [
            ('classifier_acc', classifier_acc)
        ]

        return metrics

    def transform_labels(self, labels):
        ans = torch.zeros(labels.shape[0], dataset_config.num_classifiers)
        for i, l in enumerate(labels):
            ans[i][:l] = 1

        if torch.cuda.is_available():
            ans = ans.cuda()

        return ans

    def get_class_pred(self, pred):
        prob = pred.sigmoid()

        class_pred = torch.zeros(labels.shape[0], dataset_config.num_classes)

        class_pred[:, 0] = 1 - prob[:, 0]
        for i in range(1, class_pred.shape[1] - 1):
            class_pred[:, i] = prob[:, i - 1] - prob[:, i]
        class_pred[:, -1] = prob[:, -1]

        class_pred = class_pred.argmax(dim=1)

        if torch.cuda.is_available():
            class_pred = class_pred.cuda()

        return class_pred
