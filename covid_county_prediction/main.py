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
parser.add_argument('--start-date', default='2020-01-01')
parser.add_argument('--end-date', default='2020-03-31')

args = parser.parse_args()

global_config.set_static_val('data_base_dir', args.data_dir, overwrite=True)
global_config.set_static_val('data_save_dir', args.data_save_dir, overwrite=True)

start_date  = datetime.strptime(args.start_date, '%Y-%m-%d').date()
end_date    = datetime.strptime(args.end_date, '%Y-%m-%d').date()

if args.mode == 'train':

    train_loader = DataLoader(CovidCountyDataset(start_date, end_date), batch_size=hyperparams.batch_size)
    runner = CovidRunner()

    runner.train(train_loader, hyperparams.epochs)
elif args.mode == 'save':
    saver = DataSaver()
    # saver.save_census_data()
    # saver.save_sg_social_distancing(start_date, end_date)
    saver.save_sg_patterns_monthly(start_date, end_date)
