import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from abc import abstractmethod, ABCMeta
import deepcovidnet.utils as utils
import time
import deepcovidnet.config.BaseRunnerConfig as config
import deepcovidnet.config.model_hyperparam_config as hyperparams
import os
from numpy import sign
import torch.optim.lr_scheduler as lr_scheduler
import warnings
import logging
from torch.utils.tensorboard import SummaryWriter
import signal
import sys


class BaseRunner(metaclass=ABCMeta):
    # inspired by https://github.com/pytorch/examples/blob/master/imagenet/main.py

    def __init__(
        self, nets, loss_fn, optimizers, best_metric_name,
        should_minimize_best_metric, exp_name, load_paths=None
    ):
        assert isinstance(nets, list), 'nets must be a list'
        assert isinstance(optimizers, list), 'optimizers must be a list'

        self.writer = SummaryWriter(config.get_tensorboard_dir(exp_name))
        self.nets   = nets
        self.name   = self.__class__.__name__
        self.best_metric_name = best_metric_name
        self.best_compare = -1 if should_minimize_best_metric else 1
        self.best_metric_val = - self.best_compare * 100000
        self.best_meter = utils.AverageMeter('best_metric')
        self.loss_fn = loss_fn
        self.optimizers = optimizers
        self.exp_name = exp_name
        self.lr_schedulers = \
            [lr_scheduler.StepLR(optimizers[i], hyperparams.lr_decay_step_size, hyperparams.lr_decay_factor)
                for i in range(len(self.optimizers))]
        self.global_step = 0

        if load_paths is not None:
            for i in range(len(load_paths)):
                if load_paths[i]:
                    self.load_model(self.nets[i], load_paths[i])

        if(torch.cuda.is_available()):
            for i in range(len(self.nets)):
                self.nets[i] = self.nets[i].cuda()

            loss_fn = loss_fn.cuda()

    def load_model(self, model, path, load_hyperparams=True):
        d = torch.load(path)
        model.load_state_dict(d['state_dict'])
        logging.info('Loading ' + d['arch'] + ' where ' + \
            d['best_metric_name'] + ' was ' + \
            str(d['best_metric_val']) + '...')
        if(d['best_metric_name'] == self.best_metric_name):
            self.best_metric_val = d['best_metric_val']

        if load_hyperparams and 'hyperparams' in d:
            hyperparams.load(d['hyperparams'])
            logging.info(f'Loading hyperparams: {d["hyperparams"]}')

    def output_weight_distribution(self, name_prefix="training_weights"):
        for net in self.nets:
            for param_name, param_val in net.named_parameters():
                if param_val.grad is not None:
                    param_distribution_tag = f'{net.__class__.__name__}/{name_prefix}/{param_name}'
                    self.writer.add_histogram(param_distribution_tag, param_val, global_step=self.global_step)

    def output_gradient_distributions(self, name_prefix="training_gradients"):
        for net in self.nets:
            for param_name, param in net.named_parameters():
                if param.grad is not None:
                    param_distribution_tag = f'{net.__class__.__name__}/{name_prefix}/{param_name}'
                    self.writer.add_histogram(param_distribution_tag, param.grad, global_step=self.global_step)

    def output_gradient_norms(self, name_prefix="training_gradient_norms"):
        for net in self.nets:
            for param_name, param in net.named_parameters():
                if param.grad is not None:
                    param_distribution_tag = f'{net.__class__.__name__}/{name_prefix}/{param_name}'
                    self.writer.add_scalar(param_distribution_tag, torch.norm(param.grad), global_step=self.global_step)

    def output_weight_norms(self, name_prefix="training_weight_norms"):
        for net in self.nets:
            for param_name, param in net.named_parameters():
                param_distribution_tag = f'{net.__class__.__name__}/{name_prefix}/{param_name}'
                self.writer.add_scalar(param_distribution_tag, torch.norm(param), global_step=self.global_step)

    def run(self, data_loader, prefix, epoch, metrics_calc):
        batch_time_meter = utils.AverageMeter('Time')
        data_time_meter  = utils.AverageMeter('Data')
        other_meters = []

        progress_display_made = False
        start_time = time.time()

        for i, batch in enumerate(data_loader):
            batch_number = epoch * len(data_loader) + i + 1
            data_time_meter.update(time.time() - start_time, n=self.get_batch_size(batch))

            # transfer from CPU -> GPU asynchronously if at all
            if torch.cuda.is_available():
                if (not isinstance(batch, list)) and (not isinstance(batch, dict)):
                    batch = batch.cuda(non_blocking=True)
                elif isinstance(batch, list):
                    for j in range(len(batch)):
                        batch[j] = batch[j].cuda(non_blocking=True)
                else:  # isinstance(batch, dict)
                    for key in batch.keys():
                        batch[key] = batch[key].cuda(non_blocking=True)

            metrics = metrics_calc(batch)

            # loss.backward is called in metrics_calc
            if metrics is not None:
                for j, (metric_name, metric_val) in enumerate(metrics):
                    if metric_val == metric_val:  # if metric_val != nan/inf
                        self.writer.add_scalar(
                            os.path.join(self.name, prefix + '_' + metric_name),
                            metric_val, self.global_step
                        )

                    if not progress_display_made:
                        other_meters.append(utils.AverageMeter(metric_name))
                        other_meters[j].update(metric_val, n=self.get_batch_size(batch))
                    else:
                        other_meters[j].update(metric_val, n=self.get_batch_size(batch))

                self.global_step += 1

                if not progress_display_made:
                    progress = utils.ProgressMeter(len(data_loader), other_meters + \
                        [batch_time_meter, data_time_meter], prefix=prefix)
                    progress_display_made = True
            elif not progress_display_made:
                progress = utils.ProgressMeter(len(data_loader), [batch_time_meter, data_time_meter], prefix=prefix)

            batch_time_meter.update(time.time() - start_time, n=self.get_batch_size(batch))
            start_time = time.time()

            if i % config.print_freq == 0:
                progress.display(i, epoch)

        if i % config.print_freq != 0:
            progress.display(i + 1, epoch)

    def train(self, train_loader, val_loader=None, validate_on_train=False):
        assert val_loader is None or not validate_on_train

        # add signal handlers
        orig_sigint_handler = signal.getsignal(signal.SIGINT)
        orig_sigterm_handler = signal.getsignal(signal.SIGTERM)

        signal.signal(signal.SIGINT, self.train_end)
        signal.signal(signal.SIGTERM, self.train_end)

        self.output_weight_distribution("weight_initializations")

        for i in range(len(self.nets)):
            self.nets[i].train()

        train_metrics_calc = self.train_batch_and_track_metrics if validate_on_train else self.train_batch_and_get_metrics

        bad_epochs = 0
        for epoch in range(hyperparams.epochs):
            self.best_meter.reset()
            self.run(train_loader, 'train', epoch, train_metrics_calc)

            if val_loader is not None:
                self.best_meter.reset()
                self.test(val_loader, validate=True)

            if val_loader is not None or validate_on_train:
                if(self.best_meter.avg == self.best_metric_val or sign(self.best_meter.avg - self.best_metric_val) == self.best_compare):
                    if self.best_meter.avg >= config.min_save_acc:
                        self.save_nets(epoch)
                    self.best_metric_val = self.best_meter.avg
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= hyperparams.early_stopping_num:
                        break

            elif epoch % config.save_freq == 0 or epoch == (hyperparams.epochs - 1):
                self.save_nets(epoch)

            for i in range(len(self.lr_schedulers)):
                if(
                    min(self.lr_schedulers[i].get_lr()) >=
                    hyperparams.min_learning_rate
                ):
                    self.lr_schedulers[i].step()

        self.train_end()
        signal.signal(signal.SIGINT, orig_sigint_handler)
        signal.signal(signal.SIGTERM, orig_sigterm_handler)

    def test(self, test_loader, validate=False):
        for i in range(len(self.nets)):
            self.nets[i].eval()

        with torch.no_grad():
            if validate:
                self.run(test_loader, 'val', 1, self.validate_batch_and_get_metrics)
            else:
                self.run(test_loader, 'test', 1, self.test_batch_and_get_metrics)

    def train_end(self, *args, **kwargs):
        # self.output_weight_distribution("final_weights")
        self.writer.add_hparams(
            hparam_dict=hyperparams.get_val_dict(),
            metric_dict={self.best_metric_name: self.best_metric_val}
        )

        if len(args):  # must have been called from sig handler
            sys.exit(0)

    def validate_batch_and_get_metrics(self, batch):
        return self.get_metrics_and_track_best(batch, self.test_batch_and_get_metrics)

    def train_batch_and_track_metrics(self, batch):
        return self.get_metrics_and_track_best(batch, self.train_batch_and_get_metrics)

    def get_metrics_and_track_best(self, batch, metrics_calc):
        metrics = metrics_calc(batch)
        did_find_name = False
        for (metric_name, metric_val) in metrics:
            if metric_name == self.best_metric_name:
                self.best_meter.update(metric_val, n=self.get_batch_size(batch))
                did_find_name = True
                break

        if not did_find_name:
            raise Exception('''Invalid best_metric_name set - 
                best_metric_name must be one of metrics
                best_metric_name: {}
                metric names: {}'''.format(self.best_metric_name, \
                [x[0] for x in metrics])
            )
        return metrics

    def save_nets(self, epoch):
        for i in range(len(self.nets)):
            torch.save({
                'arch': self.nets[i].__class__.__name__ + '_' + self.exp_name,
                'state_dict': self.nets[i].state_dict(),
                'best_metric_val': self.best_meter.avg,
                'best_metric_name': self.best_metric_name,
                'hyperparams': hyperparams.get_val_dict()
                }, os.path.join(config.models_base_dir,
                    self.nets[i].__class__.__name__ + '_' + self.exp_name + '_' + \
                    'checkpoint_' + str(epoch + 1) + '.pth')
            )

    @abstractmethod
    def train_batch_and_get_metrics(self, batch):
        '''Perform forward and backward pass here. Also perform the actual 
            update by doing optimizer.step() (remember to do 
            optimizer.zero_grad()).  Finally, use a learning rate scheduler
            (default choice can be torch.optim.lr_scheduler.StepLR)

            Return: metrics - [(metric_name, metric_val (should be scalar))]
        '''
        return

    @abstractmethod
    def test_batch_and_get_metrics(self, batch):
        '''Perform forward pass here.

            Return: metrics - [(metric_name, metric_val (should be scalar))]'''
        return

    @abstractmethod
    def get_batch_size(self, batch):
        return
