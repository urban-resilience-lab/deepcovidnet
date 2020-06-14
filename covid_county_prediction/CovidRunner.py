from covid_county_prediction.CovidModule import CovidModule
from covid_county_prediction.BaseRunner import BaseRunner
import torch
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import torch.nn as nn
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config


def _check_no_nan_input(f):
    def wrapper(s, batch_dict):
        for k in batch_dict:
            if k != dataset_config.labels_key:
                assert (batch_dict[k] == batch_dict[k]).all(), f'{k} has nan/inf elements'

        return f(s, batch_dict)

    return wrapper


class OrdinalBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(OrdinalBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, labels):
        return self.bce_loss(
                pred.flatten().unsqueeze(1),
                labels.flatten().unsqueeze(1),
            )


class CovidRunner(BaseRunner):
    def __init__(self, exp_name, load_path=None, sample_batch=None):
        net = CovidModule()
        self.is_optimizer_set = False

        if torch.cuda.is_available():
            net = net.cuda()
            if sample_batch:
                for k in sample_batch:
                    sample_batch[k] = sample_batch[k].cuda()

        optimizer = self.get_optimizer(net.parameters())

        hparams_dict = hyperparams.get_hparams_dict()
        hparams_dict['optim_name'] = optimizer.__class__.__name__

        if sample_batch:
            net(sample_batch)  # forward pass to set embedding module
            optimizer = self.get_optimizer(net.parameters())
            self.is_optimizer_set = True

        super(CovidRunner, self).__init__(
            nets=[net],
            loss_fn=OrdinalBCEWithLogitsLoss(),
            optimizers=[optimizer],
            best_metric_name='loss',
            should_minimize_best_metric=True,
            exp_name=exp_name,
            load_paths=[load_path],
            hparams_dict=hparams_dict
        )

        if sample_batch:
            self.writer.add_graph(self.nets[0], sample_batch)

    def get_metrics(self, pred, labels, get_loss=True):
        ordinal_labels = self._make_ordinal_labels(labels)
        loss = self.loss_fn(pred, ordinal_labels)
        acc  = self._get_accuracy(pred, labels)

        class_preds = pred.sigmoid().round()

        class_preds_mean = class_preds.float().mean().item()
        class_preds_std  = class_preds.float().std().item()

        soi_mean = self.nets[0].deep_fm.second_order_interactions.mean().item()
        soi_std  = self.nets[0].deep_fm.second_order_interactions.std().item()

        metrics = [
            ('loss', loss.mean().item()),
            ('acc', acc),
            ('class_preds_mean', class_preds_mean),
            ('class_preds_std', class_preds_std),
            ('soi_mean', soi_mean),
            ('soi_std', soi_std)
        ]

        if get_loss:
            return loss, metrics
        else:
            return metrics

    def _make_ordinal_labels(self, labels):
        ans = torch.zeros(labels.shape[0], dataset_config.num_classifiers)
        for i, l in enumerate(labels):
            ans[i][:l] = 1

        if torch.cuda.is_available():
            ans = ans.cuda()

        return ans

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
        prob = pred.sigmoid()

        class_preds = torch.zeros(labels.shape[0], dataset_config.num_classes)
        class_preds[:, 0] = 1 - prob[:, 0]
        for i in range(1, class_preds.shape[1] - 1):
            class_preds[:, i] = prob[:, i - 1] - prob[:, i]
        class_preds[:, -1] = prob[:, -1]

        class_preds = class_preds.argmax(dim=1)

        return (class_preds == labels).sum().item() / labels.shape[0]

    def get_batch_size(self, batch):
        return batch[dataset_config.labels_key].shape[0]
