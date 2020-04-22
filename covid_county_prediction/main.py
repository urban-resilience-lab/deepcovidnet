from covid_county_prediction.CovidRunner import CovidRunner
from covid_county_prediction.CovidCountyDataset import CovidCountyDataset
from covid_county_prediction.DataSaver import DataSaver
import covid_county_prediction.config.global_config as global_config
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
import logging


logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', choices=['train', 'save'])
parser.add_argument('--data-dir', default=global_config.data_base_dir)
parser.add_argument('--data-save-dir', default=global_config.data_save_dir)
parser.add_argument('--data-start-date', default='2020-02-01')
parser.add_argument('--data-end-date', default='2020-03-31')

args = parser.parse_args()

global_config.set_static_val('data_base_dir', args.data_dir, overwrite=True)
global_config.set_static_val('data_save_dir', args.data_save_dir, overwrite=True)

if args.mode == 'train':
    start_date  = datetime.strptime(args.data_start_date, '%Y-%m-%d').date
    end_date    = datetime.strptime(args.data_end_date, '%Y-%m-%d').date

    train_loader = DataLoader(CovidCountyDataset(start_date, end_date))
    runner = CovidRunner()

    runner.train(train_loader, hyperparams.epochs)
elif args.mode == 'save':
    saver = DataSaver()
    saver.save_census_data()
    saver.save_sg_social_distancing()
    # saver.save_sg_patterns_monthly()
