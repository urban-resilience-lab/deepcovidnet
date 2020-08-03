from deepcovidnet.CovidCountyDataset import CovidCountyDataset


class HypotheticalDataset(CovidCountyDataset):
    def __init__(
        self, change_factor, data_start_date,
        data_end_date, means_stds, use_cache=True,
        load_features=False
    ):
        self.change_factor = change_factor
        super(HypotheticalDataset, self).__init__(
            data_start_date, data_end_date, means_stds,
            use_cache, load_features
        )

    def create_hypothetical_features(self, features):
        self.feature_names = {f.feature_name: i for i, f in enumerate(features)}
        return self.create_hypothetical(features)

    def create_hypothetical(self, features):
        raise NotImplementedError()


class HypotheticalHomeDwellTimeDataset(HypotheticalDataset):
    def __init__(
        self, change_factor, data_start_date,
        data_end_date, means_stds, use_cache=True,
        load_features=False
    ):
        super(HypotheticalHomeDwellTimeDataset, self).__init__(
            change_factor, data_start_date, data_end_date, means_stds,
            use_cache, load_features
        )

    def create_hypothetical(self, features):
        idx = self.feature_names['social_distancing']
        for i in range(len(features[idx].raw_features)):
            s = features[idx].raw_features[i]['median_home_dwell_time']
            s *= self.change_factor
            s.clip(upper=24*60)

        return features


class HypotheticalMobilityDataset(HypotheticalDataset):
    def __init__(
        self, change_factor, data_start_date,
        data_end_date, means_stds, use_cache=True,
        load_features=False
    ):
        super(HypotheticalMobilityDataset, self).__init__(
            change_factor, data_start_date, data_end_date, means_stds,
            use_cache, load_features
        )

    def create_hypothetical(self, features):
        idx = self.feature_names['countywise_mobility']
        for i in range(len(features[idx].raw_features)):
            features[idx].raw_features[i] *= self.change_factor

        return features
