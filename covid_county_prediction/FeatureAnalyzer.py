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
    TIME = auto()
    SOI = auto()


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
                            batch[k][:, ftr_idx] = self.randomize_feature(orig_vals.shape)
                        elif batch[k].dim() == 3:
                            batch[k][:, :, ftr_idx] = self.randomize_feature(orig_vals.shape)
                        elif batch[k].dim() == 4:
                            batch[k][:, :, :, ftr_idx] = self.randomize_feature(orig_vals.shape)

                        # find new acc and keep its track
                        self.track_acc(batch, ftr_idx, ftr_idx_to_perf, batch[k].shape[0])

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
                    batch[k] = self.randomize_feature(batch[k].shape)

                    self.track_acc(batch, ftr_idx, ftr_idx_to_perf, batch[k].shape[0])

                # get difference from the best model
                self.__features.append(
                    [k, self.__orig_acc - ftr_idx_to_perf[ftr_idx].avg]
                )
        elif analysis_type == AnalysisType.TIME:
            timesteps = None
            time_dep_keys = []
            for k in keys:
                if sample_batch[k].dim() == 3:  # time dependent features
                    time_dep_keys.append(k)
                    if timesteps is None:
                        timesteps = sample_batch[k].shape[1]
                    else:
                        assert timesteps == sample_batch[k].shape[1]

            for time_idx in tqdm(range(timesteps)):
                time_idx_to_perf = {}
                for batch in self.val_loader:
                    batch_size = None
                    for k in time_dep_keys:
                        shape = batch[k][:, time_idx, :].shape
                        batch[k][:, time_idx, :] = self.randomize_feature(shape)
                        if batch_size is None:
                            batch_size = batch[k].shape[0]
                        else:
                            assert batch_size == batch[k].shape[0]

                    self.track_acc(batch, time_idx, time_idx_to_perf, batch_size)

                self.__features.append(
                    [f'day_{time_idx + 1}', self.__orig_acc - time_idx_to_perf[time_idx].avg]
                )
        elif analysis_type == AnalysisType.SOI:
            def get_metrics(self, soi_idx, batch):
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()

                net = self.runner.nets[0]
                net.eval()
                labels = batch.pop(dataset_config.labels_key)

                emb = net.embedding_module(batch)
                net.deep_fm.compute_soi(emb)
                net.deep_fm.so_int[:, soi_idx] = 0
                pred = net.deep_fm.compute_deep(emb)
                return self.runner.get_metrics(pred, labels, get_loss=False)

            net = self.runner.nets[0]
            ftr_idx_to_perf = {}
            for soi_idx in tqdm(net.deep_fm.so_int_labels):
                for batch in self.val_loader:
                    self.track_acc(
                        batch, soi_idx, ftr_idx_to_perf,
                        batch[list(batch.keys())[0]].shape[0],
                        get_metrics=lambda batch: get_metrics(self, soi_idx, batch)
                    )

                self.__features.append([
                    ' | '.join(net.deep_fm.so_int_labels[soi_idx]),
                    self.__orig_acc - ftr_idx_to_perf[soi_idx].avg
                ])

        # rank features
        self.__features.sort(key=lambda x: x[-1], reverse=True)

        df = pd.DataFrame(self.__features, columns=['feature', 'importance'])
        df.to_csv(config.get_ranks_file(self.runner.exp_name), index=False)

        return df

    def track_acc(
        self, batch, ftr_idx, ftr_idx_to_perf, batch_size, get_metrics=None
    ):
        acc = self._test_batch_and_get_acc(batch, get_metrics=get_metrics)
        if ftr_idx in ftr_idx_to_perf:
            ftr_idx_to_perf[ftr_idx].update(acc, n=batch_size)
        else:
            ftr_idx_to_perf[ftr_idx] = AverageMeter('')
            ftr_idx_to_perf[ftr_idx].update(acc, n=batch_size)

    def randomize_feature(self, shape):
        return torch.normal(mean=0, std=1, size=shape)

    def _test_batch_and_get_acc(self, batch_dict, get_metrics=None):
        if get_metrics is None:
            get_metrics = self.runner.test_batch_and_get_metrics

        for i in range(len(self.runner.nets)):
            self.runner.nets[i].eval()

        with torch.no_grad():
            labels = batch_dict[dataset_config.labels_key]
            metrics = get_metrics(batch_dict)
            batch_dict.update({dataset_config.labels_key: labels})
            for (name, val) in metrics:
                if name == self.__acc_name:
                    return val

            raise Exception(f'{self.__acc_name} not found in metrics')
