import plotly.figure_factory as ff
from datetime import timedelta
import covid_county_prediction.config.VisualizerConfig as config
import pickle
# from torch.utils.data import DataLoader
import torch
from covid_county_prediction.CovidCountyDataset import CovidCountyDataset
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import matplotlib as mpl
import covid_county_prediction.config.features_config as features_config


class Visualizer():
    def __init__(self, runner):
        self.runner = runner

    def visualize_us_map(self, dt, generate_csv=False, csv_file=None):
        dataset = CovidCountyDataset(
            dt, dt + timedelta(1),
            pickle.load(open(config.training_mean_std_file, 'rb')),
            use_cache=False
        )

        class_pred = {}
        labels = {}
        for i in range(len(dataset)):
            for k in dataset[i]:
                if k != dataset_config.labels_key:
                    dataset[i][k] = dataset[i][k].unsqueeze(0)

            fips = dataset.get_county_fips(i)

            labels[fips] = dataset[i].pop(dataset_config.labels_key)

            with torch.no_grad():
                class_pred[fips] = self.runner.get_class_pred(
                                        self.runner.nets[0](dataset[i])
                                    ).item()

        # pred_fig = ff.create_choropleth(
        #                 fips=list(class_pred.keys()),
        #                 values=list(class_pred.values()),
        #                 county_outline={'color': 'rgb(255,255,255)', 'width': 0.2}
        #             )

        # labels_fig = ff.create_choropleth(
        #                 fips=list(labels.keys()),
        #                 values=list(labels.values()),
        #                 county_outline={'color': 'rgb(255,255,255)', 'width': 0.2}
        #             )

        vs = [abs(labels[k] - class_pred[k]) for k in class_pred.keys()]
        acc = round((np.array(vs) == 0).sum() * 100 / len(vs), 1)
        diff_fig = ff.create_choropleth(
                    fips=list(class_pred.keys()),
                    values=vs,
                    county_outline={'color': 'rgb(255,255,255)', 'width': 0.1},
                    title=dict(
                        text=f'{dt.strftime("%B, %d")}. Accuracy: {acc}%',
                        x=0.5
                    ),
                    colorscale=['#388697', '#6FD08C', '#FAC05E', '#EF6461'],
                    legend_title='Class Difference',
                    plot_bgcolor='#FFFFFF',
                    font=dict(family='arial'),
                    legend=dict(font=dict(size=15), x=0.92),
                )

        # for i in range(len(pred_fig.data)):
        #     if pred_fig.data[i]['name']:
        #         try:
        #             pred_fig.data[i]['name'] = \
        #                 dataset_config.label_to_str_range[int(pred_fig.data[i]['name'])]
        #             labels_fig.data[i]['name'] = \
        #                 dataset_config.label_to_str_range[int(labels_fig.data[i]['name'])]
        #         except:
        #             pass

        if generate_csv:
            df = pd.DataFrame(data={
                    'fips': list(class_pred.keys()),
                    'pred': list(class_pred.values())
                }).set_index('fips').sort_index()

            df.to_csv(csv_file)

        return diff_fig

    def visualize_time_series(self, fips_codes, start_date, end_date):
        dataset = CovidCountyDataset(
            start_date, end_date,
            pickle.load(open(config.training_mean_std_file, 'rb')),
            use_cache=False
        )

        # design taken from https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
        plt.figure(figsize=(13, 13))

        # decorations
        plt.tick_params(axis="both", which="both", bottom=False, top=False,
            labelbottom=True, left=False, right=False, labelleft=True)

        # Lighten borders
        plt.gca().spines["top"].set_alpha(.3)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(.3)
        plt.gca().spines["left"].set_alpha(.3)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B %d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))

        mpl.rc('font', family='Arial')

        for fips in fips_codes:
            in_tensors = dataset.get_input_data_for(
                            fips,
                            discrete_labels=False
                        )
            labels_disc = \
                dataset.get_input_data_for(fips).pop(dataset_config.labels_key)

            labels_cont = in_tensors.pop(dataset_config.labels_key)
            with torch.no_grad():
                pred = self.runner.get_class_pred(
                        self.runner.nets[0](in_tensors)
                    )

            x = np.array([start_date + timedelta(i) for i in range((end_date - start_date).days)])
            c = np.array([abs((labels_disc[i] - pred[i]).item()) for i in range(len(labels_disc))])

            mpl.rc('font', family='Arial')

            # plot
            plt.plot(x, labels_cont, ':', label=f'{features_config.county_info.loc[fips].Name}, {features_config.county_info.loc[fips].State}')
            correct = plt.scatter(x[c == 0], labels_cont[c == 0], s=60, marker='o',
                                    color=(0.306, 0.349, 0.549))
            incorrect = plt.scatter(x[c != 0], labels_cont[c != 0], s=60, marker='o',
                                    color=(1, 0.745, 0.043))

        # other cosmetics
        legend = plt.legend(
                    [correct, incorrect],
                    ['predicted = actual', r'predicted $\ne$ actual'],
                    fontsize=23, loc='lower right'
                )

        plt.gca().add_artist(legend)
        plt.legend(fontsize=23, loc='upper right')
        for ytick in plt.yticks()[0]:
            plt.hlines(
                ytick, x[0], x[-1], colors='black', alpha=0.3,
                linestyles="--", lw=0.5
            )

        plt.ylabel('Rise in Cases', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        return plt.gcf()
