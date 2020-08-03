from deepcovidnet.CovidModule import CovidModule
from deepcovidnet.CovidRunner import CovidRunner
import torch
import deepcovidnet.config.model_hyperparam_config as hyperparams
import torch.nn as nn
import deepcovidnet.config.CovidCountyDatasetConfig as dataset_config


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

        eps = 0.0000001
        if loss.item() > eps:
            self.last_losses = [bce.item() / loss.item(), ce.item() / loss.item()]
        else:
            self.last_losses = [bce.item() / (loss.item() + eps), ce.item() / (loss.item() + eps)]

        return loss


class OrdinalCovidRunner(CovidRunner):
    def __init__(
        self, exp_name, net=None, loss_fn=None, load_path=None,
        sample_batch=None
    ):
        if net is None:
            net = CovidModule(output_neurons=dataset_config.num_classifiers)

        if loss_fn is None:
            loss_fn = CustomLoss()

        super(OrdinalCovidRunner, self).__init__(
            exp_name=exp_name,
            net=net,
            loss_fn=loss_fn,
            load_path=load_path,
            sample_batch=sample_batch
        )

    def _get_extra_metrics(self, pred, labels):
        metrics = [
            ('classifier_acc', self.get_classifier_acc(pred, labels)),
            ('bce_contrib', self.loss_fn.last_losses[0]),
            ('ce_contrib', self.loss_fn.last_losses[1])
        ]

        return self.get_classifier_based_acc_and_error(pred, labels) + metrics

    def get_classifier_based_acc_and_error(self, pred, labels):
        ans = []
        bin_pred = self.get_bin_class_pred(pred)
        class_pred = bin_pred.sum(dim=1)
        ans.append(('bin_acc', ((class_pred == labels).sum().float() / class_pred.numel()).item()))

        err = 0
        for i in range(bin_pred.shape[0]):
            if not (bin_pred[i, :] == bin_pred[i, :].sort(descending=True)[0]).all():
                err += 1
        ans.append(('bin_err', err / bin_pred.shape[0]))

        return ans

    def get_classifier_acc(self, pred, labels):
        ordinal_labels = get_ordinal_labels(labels).flatten()
        flat_class_pred = self.get_bin_class_pred(pred).flatten()
        return \
            (flat_class_pred == ordinal_labels).sum().item() \
            / flat_class_pred.numel()

    def get_bin_class_pred(self, pred):
        prob = pred.sigmoid()
        for i in range(prob.shape[1]):
            thresh = nn.Threshold(getattr(hyperparams, f'bin_thresh_{i}'), 0)
            prob[:, i] = thresh(prob[:, i]).ceil()
        return prob.long()

    def get_class_pred(self, pred):
        class_prob = get_class_prob(pred)
        return class_prob.argmax(dim=1)
