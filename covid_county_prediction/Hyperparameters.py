from enum import IntEnum
import logging
import covid_county_prediction.config.HyperparametersConfig as config
from ax.service.managed_loop import optimize


class HPLevel(IntEnum):
    NONE = -1
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class Hyperparameter():
    def __init__(
        self, name, val, hp_range, level=HPLevel.NONE, hp_type=float,
        check=None, log_scale=False
    ):
        self.name = name
        self.check = check
        self.range = hp_range
        self.level = level
        self.type = hp_type
        self.val = val
        self.is_log_scale = log_scale

    def add_check(self, f):
        self.check = f
        assert self.check(self.val), 'invalid value {self.val} for {self.name}'

    def __setattr__(self, name, val):
        if name == 'val':
            if self.check is not None:
                assert isinstance(val, int) or isinstance(val, str) or isinstance(val, float) or isinstance(val, bool), f'{type(val)} of {self.name} not permissible'
                assert self.check(val), 'invalid value {val} for {self.name}'
        self.__dict__[name] = val


class HyperparametersSingleton(object):

    __instance = None

    def __init__(self, add_hyperparams=None):
        pass  # everything already done in __new__

    def __new__(cls, add_hyperparams=None):
        # allows creation of only one object - singleton design pattern
        if HyperparametersSingleton.__instance is None:
            HyperparametersSingleton.__instance = \
                super(HyperparametersSingleton, cls).__new__(cls)
            HyperparametersSingleton.__instance.__hps = {}
            HyperparametersSingleton.__instance.temp_level = None

            if add_hyperparams is not None:
                add_hyperparams(HyperparametersSingleton.__instance)

        return HyperparametersSingleton.__instance

    def __call__(self, level):
        self.temp_level = level
        return self

    def __enter__(self):
        assert self.temp_level is not None, 'must pass level in "with" statement'
        return self

    def __exit__(self, type, val, tb):
        self.temp_level = None

    def __getattr__(self, name):
        if name in self.__hps:
            return self.__hps[name].val

        if name == '__name__':
            return __name__

        raise AttributeError(f'{name} does not exist')

    def add(
        self, **hparam_kwargs
    ):
        if 'level' not in hparam_kwargs or hparam_kwargs['level'] is None:
            assert self.temp_level is not None
            hparam_kwargs['level'] = self.temp_level
        if hparam_kwargs['name'] not in self.__hps:
            self.__hps[hparam_kwargs['name']] = \
                Hyperparameter(**hparam_kwargs)
        else:
            name = hparam_kwargs['name']
            logging.warning(f'Ignoring attempt to add {name} again')

    def tune(self, exp, level=HPLevel.MEDIUM):
        parameters = []
        for k in self.__hps:
            if self.__hps[k].level >= level:
                parameters.append(
                    {
                        'name': k,
                        'type': 'range',
                        'bounds': self.__hps[k].range,
                        'value_type': self.__hps[k].type.__name__,
                        'log_scale': self.__hps[k].is_log_scale
                    }
                )

        return optimize(
            parameters=parameters,
            evaluation_function=lambda config: self.__run(exp, config),
            minimize=False,
            total_trials=config.total_trials
        )

    def __run(self, experiment, parameters):
        for k in parameters:
            assert k in self.__hps
            self.__hps[k].val = parameters[k]

        experiment.train()

        return experiment.best_val

    def get_val_dict(self):
        d = {}
        for k in self.__hps:
            d[k] = self.__hps[k].val
        return d
