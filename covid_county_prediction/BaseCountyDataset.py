from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import glob
import pandas as pd
import covid_county_prediction.config.BaseCountyDatasetConfig as config
import numpy as np
import re
import os
import string
from datetime import date
import requests

### TODO: FIX TIMEZONES

class BaseCountyDataset(Dataset, ABC):
    def __init__(self):
        # load all required data here
        self.poi_info = self.get_poi_info()

    def get_poi_info(self):
        # get county code for each poi
        county_df = pd.read_csv(config.place_county_cbg_file,
                                usecols=['safegraph_place_id', 'countyFIPS'],
                                dtype={'countyFIPS': str}
                                )
        county_df = county_df.dropna().set_index('safegraph_place_id')

        # get top level category for each poi 
        cat_df = pd.DataFrame()
        for f in os.listdir(config.core_poi_path):
            if f.startswith(config.core_poi_csv_prefix):
                f = os.path.join(config.core_poi_path, f)
                temp_df = pd.read_csv(f, usecols=['safegraph_place_id', 'top_category'])
                temp_df = temp_df.dropna().set_index('safegraph_place_id')

                assert len(cat_df.index.intersection(temp_df.index)) == 0
                cat_df = pd.concat([cat_df, temp_df], axis='index')

        final_df = pd.concat([county_df, cat_df], axis='columns')

        return final_df.to_dict(orient='index')

    def make_census_dict(self):
        dfs = []
        for file in glob.glob(config.sg_open_census_data_path + "cbg_*.csv"):
            df = pd.read_csv(file, dtype={'census_block_group': str})  # The converter is used to retain leading zeros
            dfs.append(df)
        dfs = pd.concat(dfs, axis=1)
        dfs.index = dfs.iloc[:, 0].str.slice(0, 5).astype(int)
        dfs = dfs.groupby(dfs.index).sum()
        return dfs

    def read_sg_patterns_monthly(self, start_date, end_date):
        files = config.sg_patterns_monthly_reader.get_files_between(start_date, end_date)

        main_df = pd.DataFrame()

        for csv_file, month_start, month_end in files:

            index_start = month_start.day - 1
            index_end   = month_end.day - 1

            df = pd.read_csv(csv_file, 
                    usecols=[
                            'safegraph_place_id', 
                            'raw_visit_counts',
                            'visits_by_day'
                            # bucketed_dwell_times may be useful to see
                            # how long people stayed
                        ],
                    converters={'visits_by_day': (lambda x: np.array([int(s) for s in re.split(r'[,\s]\s*', x.strip('[]'))])[index_start:index_end])}
            )

            decomposed_visits_df = pd.DataFrame(
                df['visits_by_day'].values.tolist(), 
                columns=['visits_day_' + str(month_start.month).zfill(2) + '_' + str(i).zfill(2) for i in range(month_start.day, month_end.day)]
            )

            for c in decomposed_visits_df.columns:
                df[c] = decomposed_visits_df[c]

            df = df.drop(['visits_by_day'], axis=1)

            df['countyFIPS'] = df['safegraph_place_id'].apply(
                lambda x : self.poi_info[x]['countyFIPS'] 
                            if x in self.poi_info and self.poi_info[x]['countyFIPS'] 
                            else '00000'
            )

            df['top_category'] = df['safegraph_place_id'].apply(
                lambda x : self.poi_info[x]['top_category'] 
                            if x in self.poi_info and self.poi_info[x]['top_category'] 
                            else 'Unknown'
            )

            top_cats = set()
            for k in self.poi_info:
                if type(self.poi_info[k]['top_category']) == type(''):
                    top_cats.add(self.poi_info[k]['top_category'])

            for cat in top_cats:
                colname = cat.translate(str.maketrans('','',string.punctuation)).lower().replace(' ', '_')
                df[colname + '_count'] = df.apply(
                    lambda row : row['raw_visit_counts'] if cat == row['top_category'] else 0,
                    axis = 1
                )
                for i in range(month_start.day, month_end.day):
                    suffix = '_visits_day_' + str(month_start.month).zfill(2) + '_' + str(i).zfill(2) 
                    df[colname + suffix] = df.apply(
                        lambda row : row[suffix[1:]] if cat == row['top_category'] else 0,
                        axis = 1
                    )

            df = df.groupby('countyFIPS').sum()

            common_cols = main_df.columns.intersection(df.columns)

            main_df = df.merge(main_df, how='outer', suffixes=('_l', '_r'), 
                left_index=True, right_index=True)
            
            cols_to_remove = []
            for c in common_cols:
                main_df[c] = main_df[c + '_l'].add(main_df[c + '_r'], fill_value=0)
                cols_to_remove.append(c + '_l')
                cols_to_remove.append(c + '_r')

            main_df.drop(cols_to_remove, axis=1, inplace=True)

        return main_df

    def get_weather_from_fips(self, fips, start_date, end_date):
        """
        Get weather from NOAA using FIPS code (gets all relevant reportings from stations in that county)
        :param fips: Five digit FIPS code (str)
        :param start_date: start date in the form YYYY-MM-DD
        :param end_date: end date in the form YYYY-MM-DD
        :return: dictionary with TMIN (minimum temperature in tenths of a degree celcius) and TMAX (same unit max temp)
        """
        mins = []
        maxs = []
        response = requests.get(
            "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&locationid=FIPS:{}&startdate={}&enddate={}&limit=1000".format(
                str(fips), str(start_date), str(end_date)),
            headers={"token": os.environ.get("WEATHER_TOKEN")})
        data = response.json()
        for result in data.get('results'):
            datatype = result.get('datatype')
            if datatype == "TMIN":
                mins.append(result.get('value'))
            if datatype == "TMAX":
                maxs.append(result.get('value'))
        return {'TMIN': sum(mins) / len(mins), 'TMAX': sum(maxs) / len(maxs)}

    def read_sg_social_distancing(self, csv_file):
        df = pd.read_csv(csv_file, 
                usecols=[
                        'origin_census_block_group', 
                        'date_range_start', 
                        'date_range_end', 
                        'device_count',
                        'distance_traveled_from_home',
                        'completely_home_device_count',
                        'median_home_dwell_time',
                        'part_time_work_behavior_devices',
                        'full_time_work_behavior_devices'
                    ],
                dtype={'origin_census_block_group': str},
                converters={'origin_census_block_group': (lambda cbg: cbg[:5])}
            )

        #prepare for weighted average
        df['distance_traveled_from_home']   *= df['device_count']
        df['median_home_dwell_time']        *= df['device_count']

        df = df.groupby('origin_census_block_group').sum()

        df['completely_home_device_count']    /= df['device_count']
        df['part_time_work_behavior_devices'] /= df['device_count']
        df['full_time_work_behavior_devices'] /= df['device_count']
        df['distance_traveled_from_home'] /= df['device_count']
        df['median_home_dwell_time'] /= df['device_count']

        df = df.drop(['device_count'], axis=1)

        return df

    def read_num_cases(self, start_date: date, end_date: date):
        # Returns the total new cases found between start_date and end_date - 1
        df = pd.read_csv(config.labels_csv_path, usecols=[
            'date', 'fips', 'cases'
        ], dtype={'fips': str}).dropna().set_index('fips')

        start_date  = (start_date - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date    = (end_date - timedelta(days=1)).strftime('%Y-%m-%d')

        df_start = df[df['date'] == start_date].drop(['date'], axis=1)
        df_end = df[df['date'] == end_date].drop(['date'], axis=1)

        df = df_start.merge(df_end, how='inner', left_index=True, right_index=True, suffixes=('_start', '_end'))
        df['new_cases'] = df['cases_end'] - df['cases_start']
        df.drop(['cases_end', 'cases_start'], axis=1, inplace=True)

        return df

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()
