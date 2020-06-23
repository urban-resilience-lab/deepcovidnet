import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
from utils import AverageMeter
import torch
from tqdm import tqdm
from enum import Enum, auto
import pandas as pd
import covid_county_prediction.config.FeatureAnalyzerConfig as config


class AnalysisType(Enum):
    GROUP = auto()
    FEATURE = auto()


class FeatureAnalyzer():
    def __init__(self, runner, val_loader):
        self.runner = runner

        self.__acc_name = 'acc'
        assert self.runner.best_metric_name == self.__acc_name

        self.__orig_acc = self.runner.best_metric_val

        self.val_loader = val_loader
        self.__key_to_feature = self.val_loader.dataset.features.key_to_feature

        self.__features = []

    def get_ranked_features(self, analysis_type=AnalysisType.FEATURE):
        for sample_batch in self.val_loader:
            break

        keys = list(sample_batch.keys())
        keys.remove(dataset_config.labels_key)

        if analysis_type == AnalysisType.FEATURE:
            for k in tqdm(keys):
                ftr_idx_to_perf = {}
                for batch in tqdm(self.val_loader):
                    orig_shape = batch[k].shape
                    if torch.cuda.is_available():
                        batch[k] = batch[k].cuda()
                    for ftr_idx in range(batch[k].shape[-1]):
                        # randomize feature
                        orig_vals = batch[k].index_select(
                                        -1, torch.tensor(ftr_idx).cuda() if torch.cuda.is_available() else torch.tensor(ftr_idx)
                                    ).squeeze(-1)

                        if batch[k].dim() == 2:
                            batch[k][:, ftr_idx] = torch.normal(mean=0, std=1, size=orig_vals.shape)
                        elif batch[k].dim() == 3:
                            batch[k][:, :, ftr_idx] = torch.normal(mean=0, std=1, size=orig_vals.shape)
                        elif batch[k].dim() == 4:
                            batch[k][:, :, :, ftr_idx] = torch.normal(mean=0, std=1, size=orig_vals.shape)

                        # find new acc and keep its track
                        acc = self._test_batch_and_get_acc(batch)
                        if ftr_idx in ftr_idx_to_perf:
                            ftr_idx_to_perf[ftr_idx].update(acc, n=batch[k].shape[0])
                        else:
                            ftr_idx_to_perf[ftr_idx] = AverageMeter('')
                            ftr_idx_to_perf[ftr_idx].update(acc, n=batch[k].shape[0])

                        # restore feature value
                        if batch[k].dim() == 2:
                            batch[k][:, ftr_idx] = orig_vals
                        elif batch[k].dim() == 3:
                            batch[k][:, :, ftr_idx] = orig_vals
                        elif batch[k].dim() == 4:
                            batch[k][:, :, :, ftr_idx] = orig_vals

                # get difference from the best model
                for ftr_idx in ftr_idx_to_perf:
                    self.__features.append(
                        [
                            self.__key_to_feature[k].get_feature_name(ftr_idx),
                            self.__orig_acc - ftr_idx_to_perf[ftr_idx].avg
                        ]
                    )
        elif analysis_type == AnalysisType.GROUP:
            for ftr_idx, k in tqdm(enumerate(keys)):
                ftr_idx_to_perf = {}
                for batch in tqdm(self.val_loader):
                    # randomize feature
                    batch[k] = torch.normal(mean=0, std=1, size=batch[k].shape)

                    # find new acc and keep its track
                    acc = self._test_batch_and_get_acc(batch)
                    if ftr_idx in ftr_idx_to_perf:
                        ftr_idx_to_perf[ftr_idx].update(acc, n=batch[k].shape[0])
                    else:
                        ftr_idx_to_perf[ftr_idx] = AverageMeter('')
                        ftr_idx_to_perf[ftr_idx].update(acc, n=batch[k].shape[0])

                # get difference from the best model
                self.__features.append(
                    [k, self.__orig_acc - ftr_idx_to_perf[ftr_idx].avg]
                )

        # rank features
        self.__features.sort(key=lambda x: x[-1], reverse=True)

        df = pd.DataFrame(self.__features, columns=['feature', 'importance'])
        df.to_csv(config.get_ranks_file(self.runner.exp_name), index=False)

        return df

    def _test_batch_and_get_acc(self, batch_dict):
        for i in range(len(self.runner.nets)):
            self.runner.nets[i].eval()

        with torch.no_grad():
            labels = batch_dict[dataset_config.labels_key]
            metrics = self.runner.test_batch_and_get_metrics(batch_dict)
            batch_dict.update({dataset_config.labels_key: labels})
            for (name, val) in metrics:
                if name == self.__acc_name:
                    return val

            raise Exception(f'{self.__acc_name} not found in metrics')
