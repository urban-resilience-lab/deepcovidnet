from covid_county_prediction.config.base_config import Config
import sys

config = Config('Hyperparameters for DNN model')

config.epochs                       = 300

# optimizer parameters
config.lr                           = 0.05
config.momentum                     = 0.9
config.weight_decay                 = 4e-4
config.min_learning_rate            = 0.000001
config.lr_decay_step_size           = 10
config.lr_decay_factor              = 0.9

# other params
config.batch_size                   = 32
config.embedding_size               = 256
config.higher_order_features_size   = 256


sys.modules[__name__] = config
