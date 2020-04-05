from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import glob
import pandas as pd
import covid_county_prediction.constants as constants
import numpy as np
import re
import os
import string
from datetime import date

### TODO: FIX TIMEZONES

class BaseCountyDataset(Dataset, ABC):
    def __init__(self):
        # load all required data here
        self.poi_info = self.get_poi_info()

    def get_poi_info(self):
        # get county code for each poi
        county_df = pd.read_csv(constants.PLACE_COUNTY_CBG_FILE, 
                            usecols=['safegraph_place_id', 'countyFIPS'], 
                            dtype={'countyFIPS': str}
            )
        county_df = county_df.dropna().set_index('safegraph_place_id')

        # get top level category for each poi 
        cat_df = pd.DataFrame()
        for f in os.listdir(constants.CORE_POI_PATH):
            if f.startswith(constants.CORE_POI_CSV_PREFIX):
                f = os.path.join(constants.CORE_POI_PATH, f)
                temp_df = pd.read_csv(f, usecols=['safegraph_place_id', 'top_category'])
                temp_df = temp_df.dropna().set_index('safegraph_place_id')

                assert len(cat_df.index.intersection(temp_df.index)) == 0
                cat_df = pd.concat([cat_df, temp_df], axis='index')

        final_df = pd.concat([county_df, cat_df], axis='columns')

        return final_df.to_dict(orient='index')

    def make_census_dict(self):
        if __name__ == '__main__':
            main_df = pd.DataFrame()
            for file in glob.glob(constants.PATH_TO_SAFEGRAPH_OPEN_CENSUS_DATA + "cbg_*.csv"):
                df = pd.read_csv(file, converters={
                    'census_block_group': lambda x: str(x)})  # The converter is used to retain leading zeros

                # Convert census block groups to FIPS codes and use as index named FIPS
                df.index = df['census_block_group'].apply(lambda x: x[:5]).astype(
                    int)  # Save only first five chars of index
                df.drop('census_block_group', axis=1, inplace=True)
                df.index.name = "FIPS"

                # Aggregate data by FIPS codes
                df = df.groupby("FIPS").sum()
                main_df = main_df.join(df, how="outer")
            return main_df.to_dict()
    
    def read_sg_patterns(self, csv_file):
        df = pd.read_csv(csv_file, 
                usecols=[
                        'safegraph_place_id', 
                        'date_range_start', 
                        'date_range_end', 
                        'raw_visit_counts',
                        'visits_by_day'
                        # bucketed_dwell_times may be useful to see
                        # how long people stayed
                    ],
                converters={'visits_by_day': (lambda x: np.array([int(s) for s in re.split(r'[,\s]\s*', x.strip('[]'))]))}
            )
       
        if (df['date_range_start'] == df.iloc[0]['date_range_start']).all() and \
            (df['date_range_end'] == df.iloc[0]['date_range_end']).all():  
            
            start_time = df.iloc[0]['date_range_start']
            end_time   = df.iloc[0]['date_range_end']

            df.drop(labels=['date_range_start', 'date_range_end'], axis='columns', inplace=True)

        #start_time and end_time can be processed based on model here

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

        num_days = len(df.iloc[0]['visits_by_day'])

        for cat in top_cats:
            colname = cat.translate(str.maketrans('','',string.punctuation)).lower().replace(' ', '_')
            df[colname + '_count'] = df.apply(
                lambda row : row.raw_visit_counts if cat == row.top_category else 0,
                axis = 1
            )
            df[colname + '_daywise'] = df.apply(
                lambda row : row.visits_by_day if cat == row.top_category else np.zeros(num_days, dtype=int),
                axis = 1
            )

        grouped = df.groupby('countyFIPS')

        new_df = []
        for c in df.columns:
            if c not in ['safegraph_place_id', 'countyFIPS', 'top_category']:
                new_df.append(grouped[c].apply(np.sum))

        return pd.concat(new_df, axis=1)

    def read_countywise_weather(self, start_date, end_date):
        pass

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
                dtype={'origin_census_block_group': str}
            ).set_index('origin_census_block_group')

        #prepare for weighted average
        df['distance_traveled_from_home']   *= df['device_count']
        df['median_home_dwell_time']        *= df['device_count']
        
        #handle start/end dates as per model
        
        df = df.groupby(lambda cbg : str(cbg)[:5]).sum()

        df['completely_home_device_count']    /= df['device_count']
        df['part_time_work_behavior_devices'] /= df['device_count']
        df['full_time_work_behavior_devices'] /= df['device_count']
        df['distance_traveled_from_home']     /= df['device_count']
        df['median_home_dwell_time']          /= df['device_count']

        df = df.drop(['device_count'], axis=1)

        return df

    def read_num_cases(self, start_date: date, end_date: date):
        # Returns the total new cases found between start_date + 1 and end_date
        df = pd.read_csv(constants.LABELS_CSV_PATH, usecols=[
                'date', 'fips', 'cases'
            ], dtype={'fips': str}).dropna().set_index('fips')

        start_date  = start_date.strftime('%Y-%m-%d')
        end_date    = end_date.strftime('%Y-%m-%d')

        df_start = df[df['date'] == start_date].drop(['date'], axis=1)
        df_end   = df[df['date'] == end_date].drop(['date'], axis=1)

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
