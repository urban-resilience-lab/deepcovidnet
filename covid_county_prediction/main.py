import covid_county_prediction.config.global_config as global_config

from covid_county_prediction.CovidRunner import CovidRunner
from covid_county_prediction.OrdinalCovidRunner import OrdinalCovidRunner
from covid_county_prediction.CoralRunner import CoralRunner

from covid_county_prediction.CovidCountyDataset import CovidCountyDataset
from covid_county_prediction.DataSaver import DataSaver
import covid_county_prediction.config.model_hyperparam_config as hyperparams
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
from covid_county_prediction.CovidExperiment import CovidExperiment
from covid_county_prediction.FeatureAnalyzer import FeatureAnalyzer, AnalysisType
import argparse
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import pickle


def get_train_val_test_datasets(mode, use_cache=True, load_features=False):
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
            use_cache=use_cache,
            load_features=load_features
        )

        val_dataset = CovidCountyDataset(
            val_start,
            test_start,
            means_stds=train_dataset.means_stds,
            use_cache=use_cache,
            load_features=load_features
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
            use_cache=use_cache,
            load_features=load_features
        )

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_loaders(mode, load_features=False):
    assert mode in ['all', 'train', 'test']

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(mode, load_features=load_features)
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


def get_runner(runner_type):
    if runner_type == 'regular':
        return CovidRunner
    elif runner_type == 'ordinal':
        return OrdinalCovidRunner
    elif runner_type == 'coral':
        return CoralRunner


def get_analysis_type(analysis_type):
    if analysis_type == 'feature':
        return AnalysisType.FEATURE
    elif analysis_type == 'group':
        return AnalysisType.GROUP
    elif analysis_type == 'time':
        return AnalysisType.TIME


def add_args(parser):
    parser.add_argument('--exp', required=True)
    parser.add_argument('--runner', default='ordinal', choices=['regular', 'ordinal', 'coral'])
    parser.add_argument('--mode', default='train', choices=['train', 'val', 'test', 'cache', 'save', 'tune', 'rank'])
    parser.add_argument('--data-dir', default=global_config.data_base_dir)
    parser.add_argument('--data-save-dir', default=global_config.data_save_dir)
    parser.add_argument('--start-date', default=str(global_config.data_start_date))
    parser.add_argument('--end-date', default=str(global_config.data_end_date))
    parser.add_argument('--save-func', default='save_weather_data')
    parser.add_argument('--load-path', default='')
    parser.add_argument('--analysis-type', default='feature', choices=['feature', 'group', 'time'])
    parser.add_argument('--load-hps', default='')


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    global_config.set_static_val('data_base_dir', args.data_dir, overwrite=True)
    global_config.set_static_val('data_save_dir', args.data_save_dir, overwrite=True)

    if args.load_hps:
        hyperparams.load(args.load_hps)

    if args.mode == 'train':
        train_loader, val_loader, _ = get_train_val_test_loaders(args.mode)

        for b in train_loader:
            b.pop(dataset_config.labels_key)
            break  # just init b with a batch

        runner = get_runner(args.runner)(args.exp, load_path=args.load_path, sample_batch=b)

        runner.train(train_loader, val_loader=val_loader)

    elif args.mode == 'test' or args.mode == 'val':
        assert args.load_path, 'model path not specified'

        if args.mode == 'val':
            data_loader = get_train_val_test_loaders('train')[1]
        else:
            data_loader = get_train_val_test_loaders(args.mode)[2]

        for b in data_loader:
            b.pop(dataset_config.labels_key)
            break  # just init b with a batch

        runner = get_runner(args.runner)(args.exp, load_path=args.load_path, sample_batch=b)

        runner.test(data_loader)

    elif args.mode == 'cache':
        train_dataset, val_dataset, test_dataset = \
            get_train_val_test_datasets(mode='all', use_cache=False)

        train_dataset.save_cache_on_disk()
        val_dataset.save_cache_on_disk()
        test_dataset.save_cache_on_disk()

    elif args.mode == 'save':
        start_date  = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date    = datetime.strptime(args.end_date, '%Y-%m-%d').date()

        d = DataSaver()
        getattr(d, args.save_func)(start_date, end_date)

    elif args.mode == 'tune':
        train_dataset, val_dataset, _ = get_train_val_test_datasets('train')
        exp = CovidExperiment(
                args.exp,
                get_runner(args.runner),
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                exp_name=args.exp
            )

        best_params, best_vals, _, _ = hyperparams.tune(exp)

        pickle.dump(
            (best_params, best_vals),
            open(global_config.get_best_tune_file(args.exp), 'wb')
        )

    elif args.mode == 'rank':
        assert args.load_path, 'model path not specified'

        val_loader = get_train_val_test_loaders('train', load_features=True)[1]

        for b in val_loader:
            b.pop(dataset_config.labels_key)
            break

        analyzer = FeatureAnalyzer(
            runner=get_runner(args.runner)(args.exp, load_path=args.load_path, sample_batch=b),
            val_loader=val_loader
        )

        results = analyzer.get_ranked_features(
                    get_analysis_type(args.analysis_type)
                )

        print('Feature Analysis Results')
        print('=' * 80)
        print('=' * 80)
        print('=' * 80)
        print('\n' * 3)
        print(results)
        print('\n' * 3)
        print('=' * 80)
        print('=' * 80)
        print('=' * 80)
        print('\n' * 3)


if __name__ == '__main__':
    main()
