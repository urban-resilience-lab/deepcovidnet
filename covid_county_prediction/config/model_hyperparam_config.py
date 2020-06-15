from covid_county_prediction.config.base_config import Config
import sys

config = Config('Hyperparameters for DNN model')

config.epochs                       = 100

# optimizer parameters
config.lr                           = 0.0005
config.momentum                     = 0.9
config.weight_decay                 = 4e-4
config.min_learning_rate            = 0.000001
config.lr_decay_step_size           = 10
config.lr_decay_factor              = 0.9

# other params
config.batch_size                   = 64
config.embedding_size               = 128
config.higher_order_features_size   = 512
config.projection_days              = 7
config.past_days_to_consider        = 13

assert config.past_days_to_consider == 13, 'need to recache upon change'
assert config.projection_days == 7, 'need to recache upon change'

assert (config.past_days_to_consider + 1) % 7 == 0, 'to ensure features will be combinable'

def get_hparams_dict():
    hyperparams = config.__dict__.copy()
    for k in Config.static_members:
        if k in hyperparams:
            hyperparams.pop(k)
    hyperparams.pop('description')
    hyperparams.pop('get_hparams_dict')
    return hyperparams


config.get_hparams_dict = get_hparams_dict

sys.modules[__name__] = config
