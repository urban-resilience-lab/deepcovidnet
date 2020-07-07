from covid_county_prediction.DataLoader import DataLoader
from torch.utils.data import Dataset
from datetime import timedelta
from covid_county_prediction.FeaturesList import FeaturesList
import covid_county_prediction.config.RawFeatureExtractorConfig as rfe_config
import covid_county_prediction.config.CovidCountyDatasetConfig as config
import covid_county_prediction.config.features_config as features_config
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import bisect
import os
from tqdm import tqdm
import torch
from covid_county_prediction.utils import timed_logger_decorator
from math import ceil, floor
import pickle


class CovidCountyDataset(DataLoader, Dataset):
    def __init__(self, data_start_date, data_end_date, means_stds,
                 use_cache=True, load_features=False):
        super(CovidCountyDataset, self).__init__()

        self.start_date = data_start_date
        self.end_date   = data_end_date
        self.use_cache  = use_cache
        self.is_cached  = False
        self.means_stds = means_stds

        self.cache = {}

        self._load_cache_from_disk()

        if load_features or (not use_cache):
            training_data_end_date = \
                self.end_date - timedelta(hyperparams.projection_days)

            training_data_start_date = \
                self.start_date - timedelta(hyperparams.projection_days) -\
                timedelta(hyperparams.past_days_to_consider)

            features = [
                self.load_census_data(),
                self.load_pop_dens_ccvi(),
                self.load_sg_patterns_monthly(training_data_start_date, training_data_end_date),
                # self.read_weather_data(training_data_start_date, training_data_end_date),
                self.load_sg_social_distancing(training_data_start_date, training_data_end_date),
                self.load_num_cases(training_data_start_date, training_data_end_date),
                # self.load_dilation_index(training_data_start_date, training_data_end_date),
                self.load_reproduction_number(training_data_start_date, training_data_end_date),
                self.load_sg_mobility_incoming(training_data_start_date, training_data_end_date),
                self.load_countywise_cumulative_cases(training_data_start_date, training_data_end_date)
            ]

            if means_stds is None:
                means_stds = [(None, None)] * len(features)

            assert len(means_stds) == len(features)

            self.means_stds = []

            for i in range(len(features)):
                self.means_stds.append(
                    features[i].normalize(
                        mean=means_stds[i][0], std=means_stds[i][1]
                    )
                )

            self.features = FeaturesList(features)

            assert len(self.features) == config.num_features

        if self.is_cached:
            return

        self.labels_lens = []

        # load all labels
        d = self.start_date
        self.labels = []
        while d < self.end_date:
            cur_labels = self.load_num_cases(d, d + timedelta(days=1)).raw_features[0]
            cur_labels = cur_labels.dropna()

            cur_end = d - timedelta(hyperparams.projection_days)
            cur_start = cur_end - timedelta(hyperparams.past_days_to_consider)

            if cur_labels.shape[0] > 0:
                self.labels.append(
                    (
                        cur_start,
                        cur_end,
                        cur_labels
                    )
                )

                if self.labels_lens:
                    self.labels_lens.append(
                        self.labels_lens[-1] + cur_labels.shape[0]
                    )
                else:
                    self.labels_lens.append(cur_labels.shape[0])
            d += timedelta(days=1)

    def __len__(self):
        if self.is_cached:
            return len(self.cache)
        return self.labels_lens[-1]

    @timed_logger_decorator
    def _load_cache_from_disk(self):
        saved_cache_path = config.get_cached_tensors_path(
                                self.start_date, self.end_date
                            )

        if self.use_cache:
            if os.path.exists(saved_cache_path):
                self.cache = torch.load(saved_cache_path)
                self.is_cached = True
            else:
                raise Exception(f'use_cache is True but {saved_cache_path} is absent')

    def _classify_label(self, label):
        assert ceil(label) == floor(label) and label == label
        return bisect.bisect_left(config.labels_class_boundaries, label)

    def save_cache_on_disk(self):
        for i in tqdm(range(len(self))):
            self[i]  # results are automatically cached

        assert len(self.cache) == len(self)  # ensure cache is filled

        save_path = config.get_cached_tensors_path(
                        self.start_date, self.end_date
                    )
        with open(save_path, 'wb') as f:
            torch.save(self.cache, f)

    def save_means_stds(self, file):
        pickle.dump(self.means_stds, open(file, 'wb'))

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        labels_idx = bisect.bisect_left(self.labels_lens, idx)
        if self.labels_lens[labels_idx] == idx:
            labels_idx += 1

        df_idx = idx - (labels_idx > 0) * self.labels_lens[labels_idx - 1]

        out = self._get_tensors(labels_idx, df_idx)

        if idx not in self.cache:
            self.cache[idx] = out

        return out

    def _get_tensors(self, labels_idx, df_idx, discrete_labels=True):
        out = self.features.extract_torch_tensors(
                county_fips=self.labels[labels_idx][2].iloc[df_idx].name,
                start_date=self.labels[labels_idx][0],
                end_date=self.labels[labels_idx][1]
            )

        if torch.cuda.is_available():
            for k in out:
                out[k] = out[k].cuda()

        out[config.labels_key] = self.labels[labels_idx][2].values[df_idx, 0]
        if discrete_labels:
            out[config.labels_key] = self._classify_label(out[config.labels_key])

        return out

    def get_county_fips(self, idx):
        labels_idx = bisect.bisect_left(self.labels_lens, idx)
        df_idx = idx - (labels_idx > 0) * self.labels_lens[labels_idx - 1]
        return self.labels[labels_idx][2].iloc[df_idx].name

    def get_input_data_for(self, fips, discrete_labels=True):
        assert fips in features_config.county_info.index

        out = None
        for labels_idx in range(len(self.labels)):
            if fips in self.labels[labels_idx][2].index:
                df_idx = self.labels[labels_idx][2].index.get_loc(fips)
                if out is None:
                    out = self._get_tensors(labels_idx, df_idx, discrete_labels)
                    for k in out:
                        if k != config.labels_key:
                            out[k] = out[k].unsqueeze(0)
                        else:
                            out[k] = torch.tensor([int(out[k])])
                else:
                    temp = self._get_tensors(labels_idx, df_idx, discrete_labels)
                    assert out.keys() == temp.keys()
                    for k in out:
                        if k != config.labels_key:
                            out[k] = torch.cat([out[k], temp[k].unsqueeze(0)])
                        else:
                            out[k] = torch.cat([out[k], torch.tensor([int(temp[k])])])

        return out
