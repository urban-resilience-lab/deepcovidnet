from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import pandas as pd
import covid_county_prediction.config.BaseCountyDatasetConfig as config
import numpy as np
import re
import os
import string
import requests
from datetime import date, timedelta
import covid_county_prediction.config.features_config as features_config
import logging
from covid_county_prediction.ConstantFeatures import ConstantFeatures
from covid_county_prediction.CountyWiseTimeDependentFeatures import CountyWiseTimeDependentFeatures
from covid_county_prediction.TimeDependentFeatures import TimeDependentFeatures

# TODO: FIX TIMEZONES
# TODO: DEALING WITH NA VALUES


class BaseCountyDataset(Dataset, ABC):
    def __init__(self):
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

    def read_census_data(self):
        main_df = pd.DataFrame()

        for f in os.listdir(config.sg_open_census_data_path):
            if f.startswith('cbg_b') or f.startswith('cbg_c'):
                f = os.path.join(config.sg_open_census_data_path, f)
                df = pd.read_csv(f, dtype={'census_block_group': str}) 
                df['census_block_group'] = df['census_block_group'].apply(lambda x : x[:5])
                df = df.groupby('census_block_group').sum()
                main_df = main_df.merge(df, how='outer', left_index=True, right_index=True, suffixes=('', ''))

        f = os.path.join(config.sg_open_census_metadata_path, 'cbg_field_descriptions.csv')
        meta_df = pd.read_csv(f, usecols=['table_id', 'field_full_name']).set_index('table_id')

        cols_dict = {}
        for idx in meta_df.index:
            cols_dict[idx] = meta_df.loc[idx]['field_full_name']

        main_df = main_df.rename(columns=cols_dict)

        cols_to_remove = [c for c in main_df.columns if 'Margin of Error' in c]
        main_df.drop(cols_to_remove, axis=1, inplace=True)

        return ConstantFeatures(main_df, 'open_census_data')

    def _get_names_starting_with(self, original_start_date, cur_start_date, cur_end_date, prefix):
        ans = []

        d = cur_start_date
        while d < cur_end_date:
            ans.append(prefix + str((d - original_start_date).days))
            d += timedelta(days=1)

        return ans

    def read_sg_patterns_monthly(self, start_date, end_date):
        files = config.sg_patterns_monthly_reader.get_files_between(start_date, end_date)

        main_df = pd.DataFrame()

        for csv_file, month_start, month_end in files:

            index_start = month_start.day - 1
            index_end   = (month_end - timedelta(1)).day

            logging.info(f'Reading {csv_file}...')
            df = pd.read_csv(csv_file, 
                    usecols=[
                            'safegraph_place_id', 
                            'visits_by_day'
                            # bucketed_dwell_times may be useful to see
                            # how long people stayed
                        ],
                    converters={'visits_by_day': (lambda x: np.array([int(s) for s in re.split(r'[,\s]\s*', x.strip('[]'))])[index_start:index_end])}
            )

            decomposed_visits_df = pd.DataFrame(
                df['visits_by_day'].values.tolist(), 
                columns=self._get_names_starting_with(start_date, month_start, month_end, 'visits_day_')
            )

            for c in decomposed_visits_df.columns:
                df[c] = decomposed_visits_df[c]

            df = df.drop(['visits_by_day'], axis=1)

            logging.info('Decomposed visits per day')
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
                for suffix in self._get_names_starting_with(start_date, month_start, month_end, '_visits_day_'):
                    df[colname + suffix] = df.apply(
                        lambda row : row[suffix[1:]] if cat == row['top_category'] else 0,
                        axis=1
                    )

            df = df.groupby('countyFIPS').sum()

            logging.info('Finished grouping by FIPS code')

            common_cols = main_df.columns.intersection(df.columns)

            main_df = df.merge(main_df, how='outer', suffixes=('_l', '_r'), 
                left_index=True, right_index=True)

            cols_to_remove = []
            for c in common_cols:
                main_df[c] = main_df[c + '_l'].add(main_df[c + '_r'], fill_value=0)
                cols_to_remove.append(c + '_l')
                cols_to_remove.append(c + '_r')

            main_df.drop(cols_to_remove, axis=1, inplace=True)
            logging.info('Finished merging columns')

        output_dfs = []
        for col_suffix in self._get_names_starting_with(start_date, start_date, end_date, 'day_'):
            cols = [c for c in main_df.columns if c.endswith(col_suffix)]
            renamed_cols = {}
            for c in cols:
                renamed_cols[c] = c[:-len(col_suffix)]
            output_dfs.append(main_df[cols].rename(columns=renamed_cols))

        return \
            TimeDependentFeatures(output_dfs, 'sg_patterns_monthly', start_date, timedelta(days=1))

    def read_weather_data(start_date, end_date):
        """
        Get weather from NOAA using FIPS code (gets all relevant reportings from stations in that county)
        :param start_date: start date in the form YYYY-MM-DD
        :param end_date: end date in the form YYYY-MM-DD
        :return: dictionary with TMIN (minimum temperature in tenths of a degree celcius) and TMAX (same unit max temp)
        """
        # TODO: Return list of output dfs
        counties = features_config.county_info

        i = 0
        for index, row in counties.iterrows():
            if i > 50:
                break
            i = i + 1
            mins = []
            maxs = []
            response = requests.get(
                "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&locationid=FIPS:{}&startdate={}&enddate={}&limit=1000".format(
                    str(index), str(start_date), str(end_date)),
                headers={"token": os.environ.get("WEATHER_TOKEN")})
            data = response.json()
            try:
                for result in data.get('results'):
                    datatype = result.get('datatype')
                    if datatype == "TMIN":
                        mins.append(result.get('value'))
                    elif datatype == "TMAX":
                        maxs.append(result.get('value'))

            except TypeError:
                pass
            try:
                counties.at[index, 'Temp Min'] = sum(mins) / len(mins)
                counties.at[index, 'Temp Max'] = sum(maxs) / len(maxs)
            except:
                pass
        return TimeDependentFeatures(counties, 'weather_data')

    def read_sg_social_distancing(self, start_date, end_date):
        output_dfs = []

        files = config.sg_social_distancing_reader.get_files_between(start_date, end_date)

        for csv_file, cur_date, _ in files:
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
                ).set_index('origin_census_block_group')

            #prepare for weighted average
            df['distance_traveled_from_home']   *= df['device_count']
            df['median_home_dwell_time']        *= df['device_count']

            df = df.groupby(lambda cbg: cbg[:5]).sum()

            df['completely_home_device_count']      /= df['device_count']
            df['part_time_work_behavior_devices']   /= df['device_count']
            df['full_time_work_behavior_devices']   /= df['device_count']
            df['distance_traveled_from_home']       /= df['device_count']
            df['median_home_dwell_time']            /= df['device_count']

            df = df.drop(['device_count'], axis=1)

            output_dfs.append(df.dropna())

        return \
            TimeDependentFeatures(output_dfs, 'sg_social_distancing', start_date, timedelta(days=1))

    def read_num_cases(self, start_date: date, end_date: date, are_labels=False):
        # Returns the total new cases found between start_date and end_date - 1
        df = pd.read_csv(config.labels_csv_path, usecols=[
            'date', 'fips', 'cases'
        ], dtype={'fips': str}).dropna().set_index('fips')

        output_dfs = []

        cur_date = start_date
        while cur_date < end_date:
            df_yesterday    = df[df['date'] == (cur_date - timedelta(days=1)).strftime('%Y-%m-%d')]
            df_today        = df[df['date'] == cur_date.strftime('%Y-%m-%d')]

            cur_df = df_yesterday.merge(df_today, how='right', left_index=True, right_index=True, suffixes=('_start', '_end'))

            cur_df['new_cases'] = cur_df['cases_end'].subtract(cur_df['cases_start'], fill_value=0)
            cur_df.drop(['cases_end', 'cases_start', 'date_end', 'date_start'], axis=1, inplace=True)

            output_dfs.append(cur_df.fillna(0))

            cur_date += timedelta(days=1)

        if are_labels:
            assert len(output_dfs) == 1
            return output_dfs[0]         
        else:
            return TimeDependentFeatures(output_dfs, 'new_cases', start_date, timedelta(days=1))

    def read_sg_mobility_incoming(self, start_date, end_date):
        files = config.sg_patterns_weekly_reader.get_files_between(start_date, end_date)

        def sum_county_dict(d):
            ans = {}
            for k, v in d.items():
                new_k = k[:5]
                if new_k in ans:
                    ans[new_k] += v
                else:
                    ans[new_k] = v
            return ans

        output_dfs = []

        for csv_file, _, _ in files:
            df = pd.read_csv(csv_file, 
                    usecols=[
                            'safegraph_place_id', 
                            'visitor_home_cbgs'
                        ],
                    converters={
                        'safegraph_place_id': (lambda x : self.poi_info[x]['countyFIPS'] if x in self.poi_info else None),
                        'visitor_home_cbgs' : (lambda x : eval(x))
                    }
            ).dropna()  # remove all rows for which safegraph_place_id does not have a county

            df = df.groupby('safegraph_place_id').agg(
                lambda series: {k: v for d in series for k, v in d.items()}
            )  # merge dictionaries

            df['visitor_home_cbgs'] = df['visitor_home_cbgs'].apply(sum_county_dict)

            mobility_df = pd.DataFrame(
                index=features_config.county_info.index, 
                columns=features_config.county_info.index
            )

            for to_county in df.index:
                for from_county, traffic in df['visitor_home_cbgs'].loc[to_county].items():
                    if to_county in mobility_df and from_county in mobility_df:
                        mobility_df.loc[to_county].loc[from_county] = traffic

            output_dfs.append(mobility_df.fillna(0))

        return CountyWiseTimeDependentFeatures(output_dfs, 'mobility_data')

    def read_countywise_num_cases(self, start_date, end_date):
        pass

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError()
