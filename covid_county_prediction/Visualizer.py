
import plotly.figure_factory as ff
from datetime import timedelta
import covid_county_prediction.config.VisualizerConfig as config
import pickle
# from torch.utils.data import DataLoader
import torch
from covid_county_prediction.CovidCountyDataset import CovidCountyDataset
import covid_county_prediction.config.CovidCountyDatasetConfig as dataset_config
import matplotlib.pyplot as plt


class Visualizer():
    def __init__(self, runner):
        self.runner = runner

    def visualize_us_map(self, dt):
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

        pred_fig = ff.create_choropleth(
                        fips=list(class_pred.keys()),
                        values=list(class_pred.values())
                    )

        labels_fig = ff.create_choropleth(
                        fips=list(labels.keys()),
                        values=list(labels.values())
                    )

        diff_fig = ff.create_choropleth(
                        fips=list(class_pred.keys()),
                        values=[abs(labels[k] - class_pred[k])
                                for k in class_pred.keys()]
                    )

        for i in range(len(pred_fig.data)):
            if pred_fig.data[i]['name']:
                try:
                    pred_fig.data[i]['name'] = \
                        dataset_config.label_to_str_range[int(pred_fig.data[i]['name'])]
                except:
                    pass

        for i in range(len(labels_fig.data)):
            if labels_fig.data[i]['name']:
                try:
                    labels_fig.data[i]['name'] = \
                        dataset_config.label_to_str_range[int(labels_fig.data[i]['name'])]
                except:
                    pass

        return pred_fig, labels_fig, diff_fig

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
