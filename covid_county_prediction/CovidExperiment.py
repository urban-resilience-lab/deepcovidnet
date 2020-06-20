from torch.utils.data import DataLoader
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import signal
import sys


class CovidExperiment():
    def __init__(
        self, name, runner_cls, train_dataset,
        val_dataset=None, test_dataset=None, **runner_args
    ):
        self.name           = name
        self.runner_cls     = runner_cls
        self.runner_args    = runner_args
        self.train_dataset  = train_dataset
        self.val_dataset    = val_dataset
        self.test_dataset   = test_dataset
        self.run_num        = 0
        self.best_val       = None

    def train(self, **train_args):
        self.runner_args['exp_name'] = f'{self.name}_{self.run_num}'
        runner = self.runner_cls(**self.runner_args)
        print(hyperparams.lr, hyperparams.batch_size)

        train_loader = DataLoader(
                            self.train_dataset,
                            batch_size=hyperparams.batch_size,
                            shuffle=True
                        )

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                            self.val_dataset,
                            batch_size=hyperparams.batch_size,
                            shuffle=False
                        )

        runner.train(train_loader, val_loader=val_loader, **train_args)
        self.best_val = runner.best_metric_val

        self.run_num += 1