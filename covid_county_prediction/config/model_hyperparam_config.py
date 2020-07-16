
from covid_county_prediction.Hyperparameters import HPLevel, HyperparametersSingleton
import sys


def add_hyperparameters(hps):
    with hps(level=HPLevel.HIGH):
        hps.add(name='lr', val=0.0002375, hp_range=[0.00001, 0.001], log_scale=True)
        hps.add(name='weight_decay', val=2.94334e-05, hp_range=[0.000001, 0.1], log_scale=True)
        hps.add(name='batch_size', val=71, hp_range=[16, 96], hp_type=int)
        hps.add(name='embedding_size', val=408, hp_range=[32, 512], hp_type=int)
        hps.add(name='higher_order_features_size', val=362, hp_range=[32, 1024], hp_type=int)
        hps.add(name='deep_intermediate_size', val=468, hp_range=[64, 512], hp_type=int)
        hps.add(name='deep_layers', val=2, hp_range=[2, 6], hp_type=int)
        hps.add(name='dropout_prob', val=0.565068, hp_range=[0, 1])
        hps.add(name='alpha_dropout_prob', val=0.83584, hp_range=[0, 1])

    with hps(level=HPLevel.MEDIUM):
        pass

    with hps(level=HPLevel.LOW):
        hps.add(name='lr_decay_factor', val=0.6, hp_range=[0.01, 1])
        hps.add(name='ce_coeff', val=1, hp_range=[0, 100])

        hps.add(name='bin_thresh_0', val=0.5, hp_range=[0, 1])
        hps.add(name='bin_thresh_1', val=0.5, hp_range=[0, 1])
        hps.add(name='bin_thresh_2', val=0.5, hp_range=[0, 1])
        hps.add(name='epochs', val=40, hp_range=[10, 100], hp_type=int)
        hps.add(name='min_learning_rate', val=0.000001, hp_range=[0, 0.01])
        hps.add(name='lr_decay_step_size', val=10, hp_range=[1, 20], hp_type=int)
        hps.add(name='early_stopping_num', val=7, hp_range=[3, 20], hp_type=int)

    with hps(level=HPLevel.NONE):
        hps.add(
            name='projection_days', val=7, hp_range=[1, 14], hp_type=int,
            check=(lambda x: x == 7)
        )

        hps.add(
            name='past_days_to_consider', val=13, hp_range=[13, 13],
            hp_type=int, check=(lambda x: x == 13 and (x + 1) % 7 == 0)
        )


sys.modules[__name__] = HyperparametersSingleton(add_hyperparameters)
