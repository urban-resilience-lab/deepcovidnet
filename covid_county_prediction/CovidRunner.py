from covid_county_prediction.CovidModule import CovidModule
from covid_county_prediction.BaseRunner import BaseRunner
import torch
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import torch.nn as nn
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
import covid_county_prediction.config.global_config as global_config


def _check_no_nan_input(f):
    def wrapper(s, batch_dict):
        for k in batch_dict:
            if k != dataset_config.labels_key:
                assert (batch_dict[k] == batch_dict[k]).all(), f'{k} has nan/inf elements'

        return f(s, batch_dict)

    return wrapper


class CovidRunner(BaseRunner):
    def __init__(self, load_path=None):
        net = CovidModule()
        self.is_optimizer_set = False

        if torch.cuda.is_available():
            net = net.cuda()

        optimizer = self.get_optimizer(
                        net.parameters()
                    )

        global_config.comet_exp.log_parameter(
            name='optim_name',
            value=optimizer.__class__.__name__
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
        class_preds_max  = class_preds.max().item()
        class_preds_min  = class_preds.min().item()

        float_labels = labels.float()

        gt_mean = float_labels.mean().item()
        gt_std  = float_labels.std().item()
        gt_max  = float_labels.max().item()
        gt_min  = float_labels.min().item()

        soi_mean = self.nets[0].deep_fm.second_order_interactions.mean().item()
        soi_std  = self.nets[0].deep_fm.second_order_interactions.std().item()

        metrics = [
            ('loss', loss.mean().item()),
            ('acc', acc),
            ('class_preds_mean', class_preds_mean),
            ('class_preds_std', class_preds_std),
            ('class_preds_max', class_preds_max),
            ('class_preds_min', class_preds_min),
            ('gt_mean', gt_mean),
            ('gt_std', gt_std),
            ('gt_max', gt_max),
            ('gt_min', gt_min),
            ('soi_mean', soi_mean),
            ('soi_std', soi_std)
        ]

        if get_loss:
            return loss, metrics
        else:
            return metrics

    def train_batch_and_get_metrics(self, batch_dict):
        # forward pass
        labels = batch_dict.pop(dataset_config.labels_key)
        pred = self.nets[0](batch_dict)

        if not self.is_optimizer_set:
            self.optimizers[0] = self.get_optimizer(
                self.nets[0].parameters()
            )  # add parameters of embedding module too
            self.is_optimizer_set = True

        # calculate metrics
        loss, metrics = self.get_metrics(pred, labels, get_loss=True)

        # backward pass
        self.optimizers[0].zero_grad()
        loss.backward()

        # update weights
        self.optimizers[0].step()

        return metrics

    def get_optimizer(self, params):
        return torch.optim.Adam(
                params,
                lr=hyperparams.lr,
                # momentum=hyperparams.momentum,
                weight_decay=hyperparams.weight_decay
            )

    def test_batch_and_get_metrics(self, batch_dict):
        for k in batch_dict:
            if torch.cuda.is_available():
                batch_dict[k] = batch_dict[k].cuda()

        labels = batch_dict.pop(dataset_config.labels_key)
        pred = self.nets[0](batch_dict)

        return self.get_metrics(pred, labels, get_loss=False)

    def _get_accuracy(self, pred, labels):
        # assert False, 'Needs Testing'
        return (pred.argmax(dim=1) == labels).sum().item() / labels.shape[0]
