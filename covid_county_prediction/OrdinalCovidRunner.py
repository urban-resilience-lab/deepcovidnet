from covid_county_prediction.CovidModule import CovidModule
from covid_county_prediction.CovidRunner import CovidRunner
import torch
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import torch.nn as nn
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config


def get_class_prob(pred):
    prob = pred.sigmoid()

    class_prob = torch.zeros(pred.shape[0], dataset_config.num_classes)

    class_prob[:, 0] = 1 - prob[:, 0]
    for i in range(1, class_prob.shape[1] - 1):
        class_prob[:, i] = prob[:, i - 1] - prob[:, i]
    class_prob[:, -1] = prob[:, -1]

    if torch.cuda.is_available():
        class_prob = class_prob.cuda()

    return class_prob


def get_ordinal_labels(labels):
    ans = torch.zeros(labels.shape[0], dataset_config.num_classifiers)
    for i, l in enumerate(labels):
        ans[i][:l] = 1

    if torch.cuda.is_available():
        ans = ans.cuda()

    return ans


class OrdinalBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(OrdinalBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, labels):
        transformed = get_ordinal_labels(labels)

        return self.bce_loss(
                pred.flatten().unsqueeze(1),
                transformed.flatten().unsqueeze(1),
            )


class OrdinalCrossEntropy(nn.Module):
    def __init__(self):
        super(OrdinalCrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        class_prob = get_class_prob(pred)
        return self.loss(class_prob, labels)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce = OrdinalBCEWithLogitsLoss()
        self.ce  = OrdinalCrossEntropy()
        self.last_losses = None

    def forward(self, pred, labels):
        bce = self.bce(pred, labels)
        ce  = self.ce(pred, labels) * hyperparams.ce_coeff

        loss = bce + ce

        self.last_losses = [bce.item() / loss.item(), ce.item() / loss.item()]

        return loss


class OrdinalCovidRunner(CovidRunner):
    def __init__(self, exp_name, load_path=None, sample_batch=None):
        net = CovidModule(output_neurons=dataset_config.num_classifiers)

        super(OrdinalCovidRunner, self).__init__(
            exp_name=exp_name,
            net=net,
            loss_fn=CustomLoss(),
            load_path=load_path,
            sample_batch=sample_batch
        )

    def _get_extra_metrics(self, pred, labels):
        ordinal_labels = get_ordinal_labels(labels)
        classifier_acc = \
            (pred.sigmoid().round().flatten() == ordinal_labels.flatten()).sum().item() / pred.numel()

        metrics = [
            ('classifier_acc', classifier_acc),
            ('bce_contrib', self.loss_fn.last_losses[0]),
            ('ce_contrib', self.loss_fn.last_losses[1])
        ]

        return metrics

    def get_class_pred(self, pred):
        class_prob = get_class_prob(pred)
        return class_prob.argmax(dim=1)