from covid_county_prediction.CovidRunner import CovidRunner
from covid_county_prediction.CovidCountyDataset import CovidCountyDataset
from covid_county_prediction.DataSaver import DataSaver
import covid_county_prediction.config.global_config as global_config
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import argparse
from torch.utils.data import DataLoader, random_split
import logging
import torch
from datetime import timedelta, datetime


logging.getLogger().setLevel(logging.DEBUG)


def get_train_val_test_datasets(start_date, end_date):
    total_days = (end_date - start_date).days

    train_days = int(total_days * global_config.train_split_pct)
    test_days  = int(total_days * global_config.test_split_pct)
    val_days   = total_days - train_days - test_days

    train_dataset = CovidCountyDataset(
        start_date,
        start_date + timedelta(train_days),
        means_stds=None
    )

    val_dataset = CovidCountyDataset(
        train_dataset.end_date,
        train_dataset.end_date + timedelta(val_days),
        means_stds=train_dataset.means_stds
    )

    test_dataset = CovidCountyDataset(
        val_dataset.end_date,
        val_dataset.end_date + timedelta(test_days),
        means_stds=train_dataset.means_stds
    )

    assert test_dataset.end_date == end_date

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_loaders(start_date, end_date):
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
                                                start_date, end_date
                                               )
    train_loader = DataLoader(
                        train_dataset,
                        batch_size=hyperparams.batch_size,
                        shuffle=True
                    )

    val_loader = DataLoader(
                    val_dataset,
                    batch_size=hyperparams.batch_size,
                    shuffle=False
                )

    test_loader = DataLoader(
                    test_dataset,
                    batch_size=hyperparams.batch_size,
                    shuffle=False
                )

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', choices=['train', 'test', 'cache', 'save'])
    parser.add_argument('--data-dir', default=global_config.data_base_dir)
    parser.add_argument('--data-save-dir', default=global_config.data_save_dir)
    parser.add_argument('--start-date', default=str(global_config.data_start_date))
    parser.add_argument('--end-date', default=str(global_config.data_end_date))
    parser.add_argument('--save-func', default='save_weather_data')
    args = parser.parse_args()

    global_config.set_static_val('data_base_dir', args.data_dir, overwrite=True)
    global_config.set_static_val('data_save_dir', args.data_save_dir, overwrite=True)

    start_date  = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date    = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    if args.mode == 'train':
        train_loader, val_loader, test_loader = get_train_val_test_loaders(
                                                    start_date, end_date
                                                )

        runner = CovidRunner()

        runner.train(train_loader, hyperparams.epochs, val_loader=val_loader)
        runner.test(test_loader)

    elif args.mode == 'test':
        test_loader = get_train_val_test_loaders()[2]

        runner = CovidRunner()

        runner.test(test_loader)

    elif args.mode == 'cache':
        train_dataset, val_dataset, test_dataset = \
            get_train_val_test_datasets(start_date, end_date)

        train_dataset.save_cache_on_disk()
        val_dataset.save_cache_on_disk()
        test_dataset.save_cache_on_disk()

    elif args.mode == 'save':
        d = DataSaver()
        getattr(d, args.save_func)(start_date, end_date)

if __name__ == '__main__':
    main()
