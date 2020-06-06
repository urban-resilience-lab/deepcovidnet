from covid_county_prediction.DataLoader import DataLoader
from torch.utils.data import Dataset
from datetime import timedelta
from covid_county_prediction.FeaturesList import FeaturesList
import covid_county_prediction.config.RawFeatureExtractorConfig as rfe_config
import covid_county_prediction.config.CovidCountyDatasetConfig as config
import bisect


class CovidCountyDataset(DataLoader, Dataset):
    def __init__(self, data_start_date, data_end_date):
        super(CovidCountyDataset, self).__init__()

        training_data_end_date   = data_end_date
        training_data_start_date = \
            data_start_date - \
            timedelta(days=rfe_config.past_days_to_consider)

        self.labels_lens = []

        # load all labels
        d = data_start_date
        self.labels = []
        while d < data_end_date:
            cur_labels = self.load_num_cases(d, d + timedelta(days=1)).raw_features[0]
            self.labels.append(
                (
                    d - timedelta(days=rfe_config.past_days_to_consider),
                    d,
                    cur_labels
                )
            )

            self.labels_lens.append(
                sum(self.labels_lens) + cur_labels.shape[0]
            )
            d += timedelta(days=1)

        self.features = FeaturesList([
            self.load_census_data(),
            self.load_sg_patterns_monthly(training_data_start_date, training_data_end_date),
            # self.read_weather_data(training_data_start_date, training_data_end_date),
            self.load_sg_social_distancing(training_data_start_date, training_data_end_date),
            self.load_num_cases(training_data_start_date, training_data_end_date),
            self.load_sg_mobility_incoming(training_data_start_date, training_data_end_date),
            self.load_countywise_cumulative_cases(training_data_start_date, training_data_end_date)
        ])

        assert len(self.features) == config.num_features

    def __len__(self):
        return self.labels_lens[-1]

    def _classify_label(self, label):
        return bisect.bisect_left(config.labels_class_boundaries, label)

    def __getitem__(self, idx):
        labels_idx = bisect.bisect_left(self.labels_lens, idx)
        if self.labels_lens[labels_idx] == idx:
            labels_idx += 1

        out = self.features.extract_torch_tensors(
                county_fips=self.labels[labels_idx][2].iloc[idx].name,
                start_date=self.labels[labels_idx][0],
                end_date=self.labels[labels_idx][1]
            )

        out[config.labels_key] = \
            self._classify_label(self.labels[labels_idx][2].iloc[idx]['new_cases'])

        return out
