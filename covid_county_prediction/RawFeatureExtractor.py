import pandas as pd
import covid_county_prediction.config.RawFeatureExtractorConfig as config
import numpy as np
import re
import os
import string
import requests
from datetime import date, timedelta
import covid_county_prediction.config.features_config as features_config
import covid_county_prediction.config.DataSaverConfig as saver_config
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import logging
from covid_county_prediction.ConstantFeatures import ConstantFeatures
from covid_county_prediction.CountyWiseTimeDependentFeatures import CountyWiseTimeDependentFeatures
from covid_county_prediction.TimeDependentFeatures import TimeDependentFeatures
import pickle


class RawFeatureExtractor():
    def __init__(self):
        self.poi_info = self.get_poi_info()

    def get_poi_info(self):
        if os.path.exists(config.poi_info_pickle_path):
            with open(config.poi_info_pickle_path, 'rb') as f:
                ans = pickle.load(f)
            return ans

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
                temp_df['top_category'] = temp_df['top_category'].apply(
                    lambda cat: cat
                                if cat in config.whitelisted_cats
                                else config.default_cat
                )
                temp_df = temp_df.dropna().set_index('safegraph_place_id')

                assert len(cat_df.index.intersection(temp_df.index)) == 0
                cat_df = pd.concat([cat_df, temp_df], axis='index')

        final_df = pd.concat([county_df, cat_df], axis='columns')

        final_dict = final_df.to_dict(orient='index')

        with open(config.poi_info_pickle_path, 'wb') as f:
            pickle.dump(final_dict, f)

        return final_dict

    def _get_names_starting_with(self, original_start_date, cur_start_date,
                                 cur_end_date, prefix):
        ans = []

        d = cur_start_date
        while d < cur_end_date:
            ans.append(prefix + str((d - original_start_date).days))
            d += timedelta(days=1)

        return ans

    def read_census_data(self):
        main_df = pd.DataFrame()

        logging.info('Read Census Data...')
        for f in os.listdir(config.sg_open_census_data_path):
            if f.startswith('cbg_b') or f.startswith('cbg_c'):
                f = os.path.join(config.sg_open_census_data_path, f)
                df = pd.read_csv(f, dtype={'census_block_group': str})
                logging.info(f'Successfully read {f}')

                df['census_block_group'] = \
                    df['census_block_group'].apply(lambda x: x[:5])
                df = df.groupby('census_block_group').sum()
                main_df = main_df.merge(df, how='outer', left_index=True,
                                        right_index=True, suffixes=('', ''))
                logging.info('Merged into main dataframe')

        f = os.path.join(config.sg_open_census_metadata_path,
                         'cbg_field_descriptions.csv')
        meta_df = pd.read_csv(f, usecols=['table_id', 'field_full_name'])\
                    .set_index('table_id')

        cols_dict = {}
        for idx in meta_df.index:
            cols_dict[idx] = meta_df.loc[idx]['field_full_name']

        main_df = main_df.rename(columns=cols_dict)

        cols_to_remove = [c for c in main_df.columns if 'Margin of Error' in c]
        main_df.drop(cols_to_remove, axis=1, inplace=True)

        svi_df = pd.read_csv(
                    config.svi_df_path,
                    usecols=['AREA_SQMI', 'E_TOTPOP', 'FIPS'],
                    dtype={'FIPS': str}
                ).set_index('FIPS')

        svi_df['Population Density'] = svi_df['E_TOTPOP'] / svi_df['AREA_SQMI']

        svi_df = svi_df[['Population Density']]

        main_df = main_df.merge(
                    svi_df, how='outer', suffixes=('', ''),
                    left_index=True, right_index=True
                )

        ccvi_df = pd.read_csv(
                    config.ccvi_csv_path,
                    dtype={'FIPS (5-digit)': str}
                ).set_index('FIPS (5-digit)')

        ccvi_df = ccvi_df.drop(
                    columns=['State', 'State Abbreviation', 'County']
                )

        main_df = main_df.merge(
                    ccvi_df, how='outer', suffixes=('', ''),
                    left_index=True, right_index=True
                )

        return ConstantFeatures(main_df, 'open_census_data',
                                feature_saver=saver_config.census_data)

    def read_sg_patterns_monthly(self, start_date, end_date):
        files = config.sg_patterns_monthly_reader.get_files_between(
                    start_date, end_date
                )

        main_df = pd.DataFrame()

        logging.info('Reading Safegraph Patterns Monthly Data')

        for csv_file, month_start, month_end in files:

            index_start = month_start.day - 1
            index_end   = (month_end - timedelta(1)).day

            df = pd.read_csv(csv_file,
                    usecols=[
                            'safegraph_place_id',
                            'visits_by_day'
                            # bucketed_dwell_times may be useful to see
                            # how long people stayed
                        ],
                    converters={'visits_by_day': (lambda x: np.array([int(s) for s in re.split(r'[,\s]\s*', x.strip('[]'))])[index_start:index_end])}
            )
            logging.info(f'Successfully read {csv_file}...')

            # decompose visits by day into different columns
            decomposed_visits_df = pd.DataFrame(
                df['visits_by_day'].values.tolist(),
                columns=self._get_names_starting_with(
                    start_date, month_start, month_end, 'visits_day_'
                )
            )

            for c in decomposed_visits_df.columns:
                df[c] = decomposed_visits_df[c]

            df = df.drop(['visits_by_day'], axis=1)

            logging.info('Decomposed visits per day')

            # find FIPS and category of poi
            df['countyFIPS'] = df['safegraph_place_id'].apply(
                lambda x: self.poi_info[x]['countyFIPS']
                            if x in self.poi_info and self.poi_info[x]['countyFIPS'] 
                            else '00000'
            )

            df['top_category'] = df['safegraph_place_id'].apply(
                lambda x: self.poi_info[x]['top_category']
                            if x in self.poi_info and self.poi_info[x]['top_category'] 
                            else 'Unknown'
            )
            logging.info('Finished getting categories')

            top_cats = set()
            for k in self.poi_info:
                if isinstance(self.poi_info[k]['top_category'], str):
                    top_cats.add(self.poi_info[k]['top_category'])

            for cat in top_cats:
                colname = cat.translate(str.maketrans('','',string.punctuation)).lower().replace(' ', '_')
                for suffix in self._get_names_starting_with(
                    start_date, month_start, month_end, '_visits_day_'
                ):
                    df[colname + suffix] = \
                        df[suffix[1:]] * (df['top_category'] == cat)

            logging.info('Finished creating category columns')

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
            TimeDependentFeatures(output_dfs, 'sg_patterns_monthly', start_date, 
                                  timedelta(days=1),
                                  feature_saver=saver_config.sg_patterns_monthly)

    def read_weather_data(self, start_date, end_date):
        county_dfs = []
        # load county data and make df with relevant data for the county
        counties = features_config.county_info
        attributes = '&datatypeid='.join(config.weather_attributes)
        for county in counties.index:
            result = requests.get(
                "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?"
                "datasetid=GHCND&locationid=FIPS:{}&startdate={}&enddate={}&limit=1000"
                "&datatypeid={}".format(
                    str(county), str(start_date), str(end_date), attributes
                ),
                headers={"token": config.weather_token}
            )

            if result.status_code != 200:
                logging.error("Unable to connect and retrieve data from NOAA. Status code:", result.status_code)
                continue

            result_json = result.json()
            if result_json:
                logging.info(f'Received data for county {county}')

                df = pd.io.json.json_normalize(result_json, 'results')
                df = df[df['datatype'].isin(config.weather_attributes)]
                df['date'] = df['date'].str[:10]
                df = df.groupby(['date', 'datatype']).agg({'value': 'mean'}).reset_index()
                df['FIPS'] = county
                county_dfs.append(df)

        # join all county data
        county_dfs = pd.concat(county_dfs, ignore_index=True)

        # filter dfs by day
        dfs_per_day = []
        dates = county_dfs['date'].drop_duplicates().sort_values()
        for d in dates:
            dfs_per_day.append(
                county_dfs[county_dfs['date'] == d].pivot(
                    index='FIPS', columns='datatype', values='value'
                )
            )

        return TimeDependentFeatures(dfs_per_day, 'weather_data',
                                     start_date, timedelta(1),
                                     feature_saver=saver_config.weather)

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

            logging.info(f'Successfully read {csv_file}')

            # prepare for weighted average
            df['distance_traveled_from_home']   *= df['device_count']
            df['median_home_dwell_time']        *= df['device_count']

            df = df.groupby(lambda cbg: cbg[:5]).sum()
            logging.info('Grouped by counties')

            df['completely_home_device_count']      /= df['device_count']
            df['part_time_work_behavior_devices']   /= df['device_count']
            df['full_time_work_behavior_devices']   /= df['device_count']
            df['distance_traveled_from_home']       /= df['device_count']
            df['median_home_dwell_time']            /= df['device_count']

            df = df.drop(['device_count'], axis=1)

            output_dfs.append(df.dropna())

        return \
            TimeDependentFeatures(output_dfs, 'sg_social_distancing',
                                  start_date, timedelta(days=1),
                                  feature_saver=saver_config.sg_social_distancing)

    def read_num_cases(self, start_date, end_date):
        df = pd.read_csv(config.labels_csv_path, usecols=[
            'date', 'fips', 'cases'
        ], dtype={'fips': str}).dropna().set_index('fips')

        output_dfs = []

        interval = timedelta(hyperparams.projection_days)

        cur_date = start_date
        while cur_date < end_date:
            df_old = df[df['date'] == str(cur_date - interval)]
            df_new = df[df['date'] == str(cur_date)]

            cur_df = df_old.merge(
                        df_new, how='right', left_index=True, right_index=True,
                        suffixes=('_start', '_end')
                    )
            cur_df['new_cases'] = cur_df['cases_end'].subtract(
                                    cur_df['cases_start'], fill_value=0
                                )
            cur_df.drop(
                ['cases_end', 'cases_start', 'date_end', 'date_start'],
                axis=1,
                inplace=True
            )

            cur_df = cur_df[cur_df['new_cases'] >= 0]  # negatives are errors

            output_dfs.append(cur_df)

            logging.info('Processed num cases for ' + str(cur_date))

            cur_date += timedelta(days=1)

        return TimeDependentFeatures(
            output_dfs, 'new_cases', start_date, timedelta(days=1),
            feature_saver=saver_config.num_cases
        )

    def read_countywise_cumulative_cases(self, start_date, end_date):
        df = pd.read_csv(config.labels_csv_path, usecols=[
            'date', 'fips', 'cases'
        ], dtype={'fips': str}).dropna().set_index('fips')

        output_dfs = []

        cur_date = start_date

        while cur_date < end_date:
            df_today = df[df['date'] == str(cur_date)]
            if df_today.shape[0]:
                output_dfs.append(
                    df_today.drop(['date'], axis=1).fillna(0)
                )
                logging.info('Processed cumulative cases for ' + str(cur_date))
            else:
                output_dfs.append(
                    pd.DataFrame(
                        0, index=df.index.drop_duplicates(),
                        columns=['cases']
                    )
                )
            cur_date += timedelta(days=1)

        return CountyWiseTimeDependentFeatures(
                output_dfs, 'countywise_new_cases', start_date,
                timedelta(days=1), cur_type='CONSTANT',
                feature_saver=saver_config.countywise_cumulative_cases
            )

    def read_sg_mobility_incoming(self, start_date, end_date):
        files = config.sg_patterns_weekly_reader.get_files_between(start_date,
                                                                   end_date)

        def merge_and_sum_dict(series):
            ans = {}
            for d in series:
                for k in d:
                    new_k = k[:5]
                    if new_k in ans:
                        ans[new_k] += d[k]
                    else:
                        ans[new_k] = d[k]

            return ans

        output_dfs = []

        for csv_file, s, _ in files:
            start_date = min(start_date, s)
            if (end_date - s).days < 7:
                continue

            # read csv & remove all rows for which safegraph_place_id does not
            # have a county
            df = pd.read_csv(csv_file,
                    usecols=[
                            'safegraph_place_id',
                            'visitor_home_cbgs'
                        ],
                    converters={
                        'safegraph_place_id': (lambda x: self.poi_info[x]['countyFIPS'] if x in self.poi_info else None),
                        'visitor_home_cbgs' : (lambda x: eval(x))
                    }
            ).rename(columns={'safegraph_place_id': 'fips'}).dropna()

            logging.info(f'Successfully read {csv_file}...')

            df = df.groupby('fips').agg(
                merge_and_sum_dict
            )

            logging.info(f'Successfully merged dictionaries...')

            mobility_df = pd.DataFrame(
                index=features_config.county_info.index,
                columns=features_config.county_info.index
            )
            mobility_df.index.name = 'fips'

            for to_county in df.index:
                if to_county in mobility_df:
                    for from_county, traffic in df['visitor_home_cbgs'].loc[to_county].items():
                        if from_county in mobility_df:
                            mobility_df.loc[to_county].loc[from_county] = traffic

            logging.info(f'Found mobility index from {csv_file}...')
            output_dfs.append(mobility_df.fillna(0))

        return CountyWiseTimeDependentFeatures(
                output_dfs, 'mobility_data', start_date, timedelta(7),
                cur_type='CROSS',
                feature_saver=saver_config.sg_mobility
            )
