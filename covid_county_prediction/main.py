import covid_county_prediction.config.global_config as global_config
from covid_county_prediction.CovidRunner import CovidRunner
from covid_county_prediction.CovidCountyDataset import CovidCountyDataset
from covid_county_prediction.DataSaver import DataSaver
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import argparse
from torch.utils.data import DataLoader, random_split
import logging
import torch
from datetime import timedelta, datetime


def get_train_val_test_datasets(mode, use_cache=True):
    assert mode in ['all', 'train', 'test']

    train_start = global_config.data_start_date
    val_start   = global_config.train_end_date
    test_start  = global_config.val_end_date
    end_date    = global_config.data_end_date

    train_dataset = None
    val_dataset   = None
    test_dataset  = None

    if mode in ['all', 'train']:
        train_dataset = CovidCountyDataset(
            train_start,
            val_start,
            means_stds=None,
            use_cache=use_cache
        )

        val_dataset = CovidCountyDataset(
            val_start,
            test_start,
            means_stds=train_dataset.means_stds,
            use_cache=use_cache
        )

    if mode in ['all', 'test']:
        means_stds = None

        if not use_cache:
            assert mode == 'all', 'mode can\'t be test when use_cache=False'
            means_stds = train_dataset.means_stds

        test_dataset = CovidCountyDataset(
            test_start,
            end_date,
            means_stds=means_stds,
            use_cache=use_cache
        )

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_loaders(mode):
    assert mode in ['all', 'train', 'test']

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(mode)
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
    logging.getLogger().setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', required=True)
    parser.add_argument('--mode', default='train', choices=['train', 'test', 'cache', 'save'])
    parser.add_argument('--data-dir', default=global_config.data_base_dir)
    parser.add_argument('--data-save-dir', default=global_config.data_save_dir)
    parser.add_argument('--start-date', default=str(global_config.data_start_date))
    parser.add_argument('--end-date', default=str(global_config.data_end_date))
    parser.add_argument('--save-func', default='save_weather_data')
    parser.add_argument('--load-path', default='')
    args = parser.parse_args()

    global_config.set_static_val('data_base_dir', args.data_dir, overwrite=True)
    global_config.set_static_val('data_save_dir', args.data_save_dir, overwrite=True)

    if args.mode == 'save':
        start_date  = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date    = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        start_date  = global_config.data_start_date
        end_date    = global_config.data_end_date

    if args.mode == 'train':
        train_loader, val_loader, _ = get_train_val_test_loaders(args.mode)

        for b in train_loader:
            break  # just init b with a batch

        runner = CovidRunner(args.exp, sample_batch=b)

        runner.train(train_loader, hyperparams.epochs, val_loader=val_loader)

    elif args.mode == 'test':
        assert args.load_path, 'model path not specified'
        test_loader = get_train_val_test_loaders(args.mode)[2]

        for b in train_loader:
            break  # just init b with a batch

        runner = CovidRunner(args.exp, load_path=args.load_path, sample_batch=b)

        runner.test(test_loader)

    elif args.mode == 'cache':
        train_dataset, val_dataset, test_dataset = \
            get_train_val_test_datasets(mode='all', use_cache=False)

        train_dataset.save_cache_on_disk()
        val_dataset.save_cache_on_disk()
        test_dataset.save_cache_on_disk()

    elif args.mode == 'save':
        d = DataSaver()
        getattr(d, args.save_func)(start_date, end_date)


if __name__ == '__main__':
    main()
