
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
        acc = round((np.array(vs) == 0).sum() * 100 / len(vs), 3)
        diff_fig = ff.create_choropleth(
                    fips=list(class_pred.keys()),
                    values=vs,
                    county_outline={'color': 'rgb(255,255,255)', 'width': 0.2},
                    title=dict(
                        text=f'{dt.strftime("%B, %d")}. Accuracy: {acc}%',
                        x=0.5
                    ),
                    colorscale=['#388697', '#6FD08C', '#FAC05E', '#EF6461'],
                    legend_title='Class Difference',
                    width=800,
                    plot_bgcolor='#FFFFFF',
                    font=dict(family='arial'),
                    legend=dict(font=dict(size=15), x=0.9),
                    autosize=True
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

    def visualize_time_series(self, fips, start_date, end_date):
        dataset = CovidCountyDataset(
            start_date, end_date,
            pickle.load(open(config.training_mean_std_file, 'rb')),
            use_cache=False
        )

        in_tensors = dataset.get_input_data_for(fips, discrete_labels=False)

        labels = in_tensors.pop(dataset_config.labels_key)
        with torch.no_grad():
            pred = self.runner.get_class_pred(self.runner.nets[0](in_tensors))

        x = [start_date + timedelta(i) for i in range((end_date - start_date).days)]
        highest = torch.max(labels).item()

        plt.fill_between(
            x=x,
            y1=[dataset_config.label_to_range[p.item()][0] for p in pred],
            y2=[dataset_config.label_to_range[p.item()][1] if p < dataset_config.num_classes - 1 else highest for p in pred],
            facecolor="orange", # The fill color
            color='blue',       # The outline color
            alpha=0.2
        )

        plt.plot(x, [label.item() for label in labels], linestyle='dashed')

        plt.show()
