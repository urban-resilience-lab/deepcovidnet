# DeepCOVIDNet: An Interpretable Deep Learning Model for Predictive Surveillance of COVID-19 Using Heterogeneous Features and Their Interactions

This is the official GitHub repository of the DeepCOVIDNet model to forecast the range of increase in the number of infected cases in all U.S. counties. For full details, refer to the original paper at https://arxiv.org/abs/2008.00115.

## Instructions to run

For full functionality, use Python 3.7+. If not available, Python 3.6 will work but will not support automatic hyperparameter tuning. 

```bash
#setup the project
python3 setup.py develop
#start working/testing now
```
Note that most interaction with this code will require changing some configuration variables in the ```deepcovidnet/config/``` folder. Of particular importance is to change the values of configuration variables [```data_base_dir```](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/config/global_config.py#L9) to a location where all raw data will be stored and [```data_save_dir```](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/config/global_config.py#L10) to a location where all saved data must be stored.

The ```data_base_dir``` directory expects certain structure inside to be easily parseable. This structure can be changed by modifying the ```deepcovidnet/deepcovidnet/config/RawFeatureExtractorConfig.py``` file, but the default expected structure is outlined below. The ```data_base_dir``` is expected to contain the following subdirectories:
 - ```core_places/```: Location of [SafeGraph Core Places data](https://docs.safegraph.com/v4.0/docs#section-core-places)
 - ```safegraph_open_census_data/```: Location of [SafeGraph Open Census Data](https://docs.safegraph.com/docs/open-census-data)
 - ```social_distancing/```: Location of [SafeGraph Social Distancing Metrics](https://docs.safegraph.com/docs/social-distancing-metrics). The data for date ```YYYY-MM-DD``` should be in ```social_distancing/YYYY/MM/DD/YYYY-MM-DD-social-distancing.csv```
 - ```monthly_patterns/```: Location of [SafeGraph Monthly Patterns](https://docs.safegraph.com/v4.0/docs/places-schema#section-patterns). The data for month ```YYYY/MM``` should be in ```monthly_patterns/YYYYMM-AllPatterns-PATTERNS-YYYY_MM/patterns-part*.csv```
 - ```weekly_patterns/```: Location of [SafeGraph Weekly Patterns](https://docs.safegraph.com/v4.0/docs/places-schema#section-patterns). The data for the week starting on ```YYYY-MM-DD``` should be in ```weekly_patterns/YYYY-MM-DD-weekly-patterns.csv```
 
For more information, refer to ```deepcovidnet/deepcovidnet/config/RawFeatureExtractorConfig.py```. Depending upon the use case, some of the above directories can be unnecessary if the user does not plan to use the features in a particular subdirectory. To modify which features to include in the model, modify [```deepcovidnet/CovidCountyDataset.py```](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/CovidCountyDataset.py#L41).
 
Some feature used in the model like Venables Distance and reproduction number were developed internally in the lab and are not available in public. Feel free to open an issue if you are interested in getting the data to understand more about the data sharing policy of the lab.
 
 ### Adding Features
 
 Including additional features is easy. To add a new feature, simply define a function in ```deepcovidnet/RawFeatureExtractor.py``` that parses raw data, a function in ```deepcovidnet/DataSaver.py``` to save the processed feature value from the raw data, and a function in ```deepcovidnet/DataLoader.py``` to load the saved feature. Finally, simply load the feature in the actual dataset [here](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/CovidCountyDataset.py#L41) to include it. Refer the example of the following three functions to understand more: [```read_sg_social_distancing```](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/RawFeatureExtractor.py#L287), [```save_sg_social_distancing```](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/DataSaver.py#L29), and [```load_sg_social_distancing```](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/DataLoader.py#L34).

### Changing Output Classes

As described in the [paper](https://arxiv.org/abs/2008.00115), currently the model predicts one of four output classes to predict the range of increase in infections in a given county. However, for some use cases, the resolution of the model would need to be made finer or coarser. To do this, one will need to change the number of output classes and the respective ranges of the increase in the number of infections. This change is simple to do and only requires changing the configuration variable [```labels_class_boundaries ```](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/config/CovidCountyDatasetConfig.py#L12) to define new end points for the output class ranges.

### Using main.py

The [deepcovidnet/main.py](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/main.py) file is designed to be solely capable of running many different types of experiments. The basic format of a command to run main.py is the following:

```
python3 main.py --exp YOUR_EXPIREMENT_NAME --mode YOUR_VALID_EXPERIMENT_MODE [any additional arguments to support the current mode]
```

where ```YOUR_EXPIREMENT_NAME``` is any string of characters used to name the current experiment which will be used in all corresponding output files (such as tensorboard logs, results, etc.) of the particular run and ```YOUR_VALID_EXPERIMENT_MODE``` is the name of one of the modes that is supported by the script. All supported modes with their functionality are described below:

 1. ```train```: This mode is used to train the model (and validate per epoch). It creates the training set using data between the dates defined by the configuration variables [```data_start_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L15) and [```train_end_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L18) and the validation set using data between dates defined by [```train_end_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L18) and [```val_end_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L19). The type of model to use for training can be defined by ```--runner``` command line argument, which defaults to the model used in the [paper](https://arxiv.org/abs/2008.00115). Since model training requires loading in many different features, it is strongly recommended to save all features and cache the training and validation datasets (as described below) before using this mode to significantly reduce computation time. To resume training from a previous point, the ```--load-path``` command line argument can be used to give the path of model from which training will start. The hyperparameters used in the model are loaded from the configuration file [```deepcovidnet/config/model_hyperparam_config.py```](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/config/model_hyperparam_config.py). By default, all models with the best accuracy on the validation set will be saved given that the accuracy is above a certain threshold defined by the configuration variable [min_save_acc](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/BaseRunnerConfig.py#L24).
 1. ```train_no_val```: This mode is exactly similar to the ```train``` mode except that it does not use a validation set to validate after every training epoch. Therefore, this mode is useful to train on the entire training and validation dataset. Similar to the previous mode, it creates the training set using data between the dates defined by the configuration variables [```data_start_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L15) and [```train_end_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L18). Since there is no validation set, the model is saved every [```save_freq```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/BaseRunnerConfig.py#L11) epochs.
 1. ```val```: In this mode, the model is only evaluated on the validation dataset created using data from the range of dates defined by [```train_end_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L18) and [```val_end_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L19). The path of the model that one wishes to perform validation on should be given by the command line argument of ```--load-path```. If a non-default/custom model is used, then ```--runner``` must also be appropriately used.
 1. ```test```: This mode evaluates the model on the testing set. The test set is created using data from the range of dates defined by [```val_end_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L19) and [```data_end_date```](https://github.com/urban-resilience-lab/deepcovidnet/blob/51024cc51d9f6dc427c56f300cb8900d63c462d7/deepcovidnet/config/global_config.py#L16). The ```--load-path``` and ```--runner``` arguments must be used appropriately as discussed in the previous point.
 1. ```cache```: This mode is used to cache the training, validation, and testing dataset on disk for faster loading in the future. The same date ranges as described above are used to define the three datasets. Caching datasets can significantly reduce training and evaluation times.
 1. ```save```: This mode is used to call functions from [```deepcovidnet/DataSaver.py```](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/DataSaver.py) to save raw features in an easily loadable format. The ```--save-func``` command line argument can be used to name a function in [```deepcovidnet/DataSaver.py```](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/DataSaver.py) that needs to be called. Further, to decide data on which range of dates should be saved, the ```--start-date``` and ```--end-date``` command line arguments should be used. Saving raw features can significantly boost the time required to cache datasets or load them in the future.
 1. ```tune```: This mode requires Python 3.7+ due to the use of [Ax](https://ax.dev/) hyperparameter optimization library. In this mode, all hyperparameters defined with importance level of ```MEDIUM``` and above in [deepcovidnet/config/model_hyperparam_config.py](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/config/model_hyperparam_config.py) will be automatically tuned using Bayesian optimization. The hyperparameters are chosen with the objective to maximize the prediction accuracy on the validation set.
 1. ```rank```: This mode is used to perform feature analysis and ranks features based on relative importance. Therefore, to use this mode, one should use the ```--load-path``` and ```--runner``` command line arguments appropriately. Further, the ```--analysis-type``` argument should be used to define whether individual features (```feature```), feature groups (```group```), time steps (```time```), or second order interactions (```soi```) should be used for importance evaluation. The analysis types in the parenthesis in the previous line show the exact value that should be passed to the  ```--analysis-type``` argument to perform the corresponding analysis.

## Citation

Please cite this work if it was helpful in your research:

```
@article{ankit2020deepcovidnet,
 title={DeepCOVIDNet: An Interpretable Deep Learning Model for Predictive Surveillance of COVID-19 Using Heterogeneous Features and Their Interactions},
 author={Ramchandani, Ankit and Fan, Chao and Mostafavi, Ali},
 journal={arXiv preprint arXiv:2008.00115},
 year={2020}
}
```

## Questions
In case of any questions, please feel free to open an issue, and we will try to answer it as soon as possible.
