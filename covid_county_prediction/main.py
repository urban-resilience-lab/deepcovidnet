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


def get_train_val_test_datasets(start_date, end_date, mode):
    assert mode in ['all', 'train', 'test']

    total_days = (end_date - start_date).days

    train_days = int(total_days * global_config.train_split_pct)
    test_days  = int(total_days * global_config.test_split_pct)
    val_days   = total_days - train_days - test_days

    train_start = start_date
    val_start   = train_start + timedelta(train_days)
    test_start  = val_start + timedelta(val_days)

    train_dataset = None
    val_dataset   = None
    test_dataset  = None

    if mode in ['all', 'train']:
        train_dataset = CovidCountyDataset(
            train_start,
            val_start,
            means_stds=None
        )

        val_dataset = CovidCountyDataset(
            val_start,
            test_start,
            means_stds=train_dataset.means_stds
        )

    if mode in ['all', 'test']:
        test_dataset = CovidCountyDataset(
            test_start,
            end_date,
            means_stds=train_dataset.means_stds
        )

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_loaders(start_date, end_date, mode):
    assert mode in ['all', 'train', 'test']

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
                                                start_date, end_date, mode
                                               )
    train_loader = None
    val_loader = None
    test_loader = None

    if mode in ['all', 'train']:
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

    if mode in ['all', 'test']:
        test_loader = DataLoader(
                        test_dataset,
                        batch_size=hyperparams.batch_size,
                        shuffle=False
                    )

    return train_loader, val_loader, test_loader


def main():
    logging.getLogger().setLevel(logging.DEBUG)

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
        train_loader, val_loader, _ = get_train_val_test_loaders(
                                        start_date, end_date, args.mode
                                      )

        runner = CovidRunner()

        runner.train(train_loader, hyperparams.epochs, val_loader=val_loader)

    elif args.mode == 'test':
        test_loader = get_train_val_test_loaders(start_date, end_date, args.mode)[2]

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
