from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config

config = Config('Hyperparameters for DNN model')

config.epochs                       = 300

# optimizer parameters
config.lr                           = 0.001
config.momentum                     = 0.9
config.weight_decay                 = 4e-4
config.min_learning_rate            = 0.000001
config.lr_decay_step_size           = 10
config.lr_decay_factor              = 0.9

# other params
config.batch_size                   = 64
config.embedding_size               = 512
config.higher_order_features_size   = 512

config.set_static_val('are_hyperparams_set', False)
if not config.are_hyperparams_set:
    hyperparams = config.__dict__
    for k in Config.static_members:
        hyperparams.pop(k)
    hyperparams.pop('description')

    global_config.comet_exp.log_parameters(hyperparams)

sys.modules[__name__] = config
