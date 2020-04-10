from covid_county_prediction.config.base_config import Config
import sys
import covid_county_prediction.config.global_config as global_config
import os
from datetime import timezone, date, timedelta
from pandas.tseries import offsets

config = Config('Config for BaseCountyDataset')

class ReaderConfig(Config):
    def __init__(self, file_granularity : str, file_path_format : str, 
                is_timezone_variable: bool, timezone: timezone = None, 
                part_prefix: str = None, file_extension='.csv'):
        
        super(ReaderConfig, self).__init__('Config for reading data from different mediums')
        
        assert file_granularity in ['daily', 'monthly']
        
        self.file_granularity   = file_granularity

        if self.file_granularity == 'daily':
            self.date_offset = timedelta(days=1)
        elif self.file_granularity == 'monthly':
            self.date_offset = offsets.MonthBegin()

        self.file_path_format   = file_path_format

        self.file_extension     = file_extension

        self.partwise = part_prefix is not None
        self.part_prefix = part_prefix

        self.is_timezone_variable = is_timezone_variable
        self.timezone = timezone

    def get_files_between(self, start_date: date, end_date: date):
        files = set()

        d = start_date
        while d < end_date:
            file_format = d.strftime(
                os.path.join(global_config.data_base_dir, self.file_path_format)
            )

            assert os.path.exists(file_format), file_format + ' does not exist'

            if os.path.isdir(file_format):
                for f in os.listdir(file_format):
                    if f.startswith(self.part_prefix) and f.endswith(self.file_extension):
                        f = os.path.join(file_format, f)
                        files.add(
                            (f, d, min(end_date, d + self.date_offset - timedelta(days=1)))
                        )
            elif os.path.isfile(file_format):
                files.add(
                    (file_format, d, min(end_date, d + self.date_offset - timedelta(days=1)))
                )

            d += self.date_offset

        return list(files)

# core poi
config.core_poi_csv_prefix = 'core_poi-part'
config.core_poi_path = os.path.join(global_config.data_base_dir, 'CoreRecords-CORE_POI-2019_03-2020-03-25')

# open census
config.sg_open_census_data_path = os.path.join(global_config.data_base_dir, "safegraph_open_census_data/data/")

# poi -> county
config.place_county_cbg_file = os.path.join(global_config.data_base_dir, 'placeCountyCBG.csv')

# labels
config.labels_csv_path = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'

# reader configs
config.sg_social_distancing_reader =  ReaderConfig(
                                    file_granularity='daily', 
                                    file_path_format='%Y/%m/%d/%Y-%m-%d-social-distancing.csv',
                                    is_timezone_variable=True
                                )

config.sg_patterns_monthly_reader = ReaderConfig(
                                        file_granularity='monthly', 
                                        file_path_format='%y%m-AllPatterns-PATTERNS-%Y_%m/',
                                        is_timezone_variable=False,
                                        timezone=timezone(timedelta()),
                                        part_prefix='patterns-part'
                                    )

config.past_days_to_consider = 20

sys.modules[__name__] = config