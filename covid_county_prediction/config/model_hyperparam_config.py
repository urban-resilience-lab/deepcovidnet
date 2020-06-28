
from covid_county_prediction.Hyperparameters import HPLevel, HyperparametersSingleton
import sys


def add_hyperparameters(hps):
    with hps(level=HPLevel.HIGH):
        hps.add(name='lr', val=0.00019327, hp_range=[0.000001, 1], log_scale=True)
        # hps.add(name='momentum', val=0.9, hp_range=[0.01, 1])
        hps.add(name='weight_decay', val=0.000019119, hp_range=[0.000001, 0.01], log_scale=True)
        hps.add(name='batch_size', val=85, hp_range=[8, 512], hp_type=int)
        hps.add(name='embedding_size', val=574, hp_range=[32, 2048], hp_type=int)
        hps.add(name='higher_order_features_size', val=550, hp_range=[32, 2048], hp_type=int)
        hps.add(name='dropout_prob', val=0.50884, hp_range=[0, 1])

    with hps(level=HPLevel.MEDIUM):
        hps.add(name='lr_decay_factor', val=0.84848, hp_range=[0.01, 1])
        hps.add(name='ce_coeff', val=16.172, hp_range=[0, 100])

    with hps(level=HPLevel.LOW):
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
