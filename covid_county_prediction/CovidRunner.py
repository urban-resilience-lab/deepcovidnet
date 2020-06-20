from covid_county_prediction.CovidModule import CovidModule
from covid_county_prediction.BaseRunner import BaseRunner
import torch
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import torch.nn as nn
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
from covid_county_prediction.Hyperparameters import HPLevel


def get_default_net():
    return CovidModule(output_neurons=dataset_config.num_classes)


class CovidRunner(BaseRunner):
    def __init__(
        self, exp_name, net=get_default_net(), loss_fn=nn.CrossEntropyLoss(),
        load_path=None, sample_batch=None
    ):
        self.is_optimizer_set = False

        if torch.cuda.is_available():
            net = net.cuda()
            if sample_batch:
                for k in sample_batch:
                    sample_batch[k] = sample_batch[k].cuda()

        optimizer = self.get_optimizer(net.parameters())

        hyperparams.add(
            name='optim_name', val=optimizer.__class__.__name__, hp_range=None,
            hp_type=str, level=HPLevel.NONE
        )

        if sample_batch:
            net(sample_batch)  # forward pass to set embedding module
            optimizer = self.get_optimizer(net.parameters())
            self.is_optimizer_set = True

        super(CovidRunner, self).__init__(
            nets=[net],
            loss_fn=loss_fn,
            optimizers=[optimizer],
            best_metric_name='acc',
            should_minimize_best_metric=False,
            exp_name=exp_name,
            load_paths=[load_path]
        )

        if sample_batch:
            self.writer.add_graph(self.nets[0], sample_batch)

    def get_metrics(self, pred, labels, get_loss=True):
        loss = self.loss_fn(pred, labels)

        class_pred = self.get_class_pred(pred)

        acc = self._get_accuracy(class_pred, labels)

        class_pred_mean = class_pred.float().mean().item()
        class_pred_std  = class_pred.float().std().item()

        soi_mean = self.nets[0].deep_fm.second_order_interactions.mean().item()
        soi_std  = self.nets[0].deep_fm.second_order_interactions.std().item()

        metrics = [
            ('loss', loss.mean().item()),
            ('acc', acc),
            ('class_preds_mean', class_pred_mean),
            ('class_preds_std', class_pred_std),
            ('soi_mean', soi_mean),
            ('soi_std', soi_std)
        ] + self.get_classwise_recall_metrics(class_pred, labels)

        metrics = metrics + self._get_extra_metrics(pred, labels)

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

    def _get_accuracy(self, class_pred, labels):
        return (class_pred == labels).sum().item() / labels.shape[0]

    def get_batch_size(self, batch):
        return batch[list(batch.keys())[0]].shape[0]

    def get_class_pred(self, pred):
        return pred.argmax(dim=1)

    def _get_extra_metrics(self, pred, labels):
        return []

    def get_classwise_recall_metrics(self, class_pred, labels):
        metrics = []
        for c in range(dataset_config.num_classes):
            tp = ((class_pred == labels) & (class_pred == c)).sum().item()
            total = (labels == c).sum().item()

            recall = tp / total if total else (1 if tp else 0)

            metrics.append((f'class_{c}_recall', recall))

        return metrics
