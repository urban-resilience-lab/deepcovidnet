from covid_county_prediction.BaseCountyDataset import BaseCountyDataset
from datetime import timedelta
from covid_county_prediction.FeaturesList import FeaturesList
import covid_county_prediction.config.BaseCountyDatasetConfig as base_county_config
import bisect


class CovidCountyDataset(BaseCountyDataset):
    def __init__(self, data_start_date, data_end_date):
        super(CovidCountyDataset, self).__init__()

        training_data_end_date   = data_end_date
        training_data_start_date = data_start_date - timedelta(days=base_county_config.past_days_to_consider)

        self.labels_lens = []

        # load all labels
        d = data_start_date
        self.labels = []
        while d < data_end_date:
            cur_labels = self.read_num_cases(d, d + timedelta(days=1), are_labels=True)
            self.labels.append(
                (
                    d - timedelta(days=base_county_config.past_days_to_consider),
                    d,
                    cur_labels
                )
            )

            self.labels_lens.append(
                sum(self.labels_lens) + cur_labels.shape[0]
            )
            d += timedelta(days=1)

        self.features = FeaturesList([
            self.read_census_data(),
            self.read_sg_patterns_monthly(training_data_start_date, training_data_end_date),
            # self.read_weather_data(training_data_start_date, training_data_end_date),
            self.read_sg_social_distancing(training_data_start_date, training_data_end_date),
            self.read_num_cases(training_data_start_date, training_data_end_date)
            # self.read_sg_mobility_incoming(training_data_start_date, training_data_end_date)
        ])

    def __len__(self):
        return self.labels_lens[-1]

    def __getitem__(self, idx):
        labels_idx = bisect.bisect_left(self.labels_lens, idx)
        if self.labels_lens[labels_idx] == idx:
            labels_idx += 1

        return \
            self.features.extract_torch_tensors(
                county_fips=self.labels[labels_idx][2].iloc[idx].name,
                start_date=self.labels[labels_idx][0],
                end_date=self.labels[labels_idx][1]
            )
