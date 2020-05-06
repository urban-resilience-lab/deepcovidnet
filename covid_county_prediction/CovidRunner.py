from covid_county_prediction.CovidModule import CovidModule
from covid_county_prediction.BaseRunner import BaseRunner
import torch
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import torch.nn as nn
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config


class CovidRunner(BaseRunner):
    def __init__(self, load_path=None):
        net = CovidModule()

        if torch.cuda.is_available():
            net = net.cuda()

        optimizer = torch.optim.SGD(
                net.parameters(),
                lr=hyperparams.lr,
                momentum=hyperparams.momentum,
                weight_decay=hyperparams.weight_decay
            )

        super(CovidRunner, self).__init__(
            models=[net],
            loss_fn=nn.CrossEntropyLoss(),
            optimizers=[optimizer],
            best_metric_name='loss',
            should_minimize_best_metric=True,
            load_paths=[load_path]
        )

    def get_metrics(self, pred, labels, get_loss=True):
        loss = self.loss_fn(pred, labels)
        acc  = self._get_accuracy(pred, labels)

        class_preds = pred.argmax(dim=1)

        class_preds_mean = class_preds.float().mean().item()
        class_preds_std  = class_preds.float().std().item()

        gt_mean = labels.float().mean().item()
        gt_std  = labels.float().std().item()

        metrics = [
            ('loss', loss.mean().item()),
            ('acc', acc),
            ('class_preds_mean', class_preds_mean),
            ('class_preds_std', class_preds_std),
            ('gt_mean', gt_mean),
            ('gt_mean', gt_std)
        ]

        if get_loss:
            return loss, metrics
        else:
            return metrics

    def train_batch_and_get_metrics(self, batch_dict):
        # forward pass
        labels = batch_dict.pop(dataset_config.labels_key)
        pred = self.nets[0](batch_dict)

        # calculate metrics
        loss, metrics = self.get_metrics(pred, labels, get_loss=True)

        # backward pass
        self.optimizers[0].zero_grad()
        loss.backward()

        # update weights
        self.optimizers[0].step()

        return metrics

    def test_batch_and_get_metrics(self, batch_dict):
        labels = batch_dict.pop(dataset_config.labels_key)
        pred = self.nets[0](batch_dict)

        return self.get_metrics(pred, labels)

    def _get_accuracy(self, pred, labels):
        # assert False, 'Needs Testing'
        return (pred.argmax(dim=1) == labels).sum().item() / labels.shape[0]
