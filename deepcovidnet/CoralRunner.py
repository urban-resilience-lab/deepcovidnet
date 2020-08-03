from deepcovidnet.CovidCoralModule import CovidCoralModule
from deepcovidnet.OrdinalCovidRunner import OrdinalCovidRunner, OrdinalBCEWithLogitsLoss


class CoralRunner(OrdinalCovidRunner):
    def __init__(self, exp_name, load_path=None, sample_batch=None):
        super(CoralRunner, self).__init__(
            exp_name,
            net=CovidCoralModule(),
            loss_fn=OrdinalBCEWithLogitsLoss(),
            load_path=load_path,
            sample_batch=sample_batch
        )

    def _get_extra_metrics(self, pred, labels):
        metrics = [
            ('classifier_acc', self.get_classifier_acc(pred, labels))
        ]

        return metrics
