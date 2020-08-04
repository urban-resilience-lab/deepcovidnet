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

### Change Output Classes

As described in the [paper](https://arxiv.org/abs/2008.00115), currently the model predicts one of four output classes to predict the range of increase in infections in a given county. However, for some use cases, the resolution of the model would need to be made finer or coarser. To do this, one will need to change the number of output classes and the respective ranges of the increase in the number of infections. This change is simple to do and only requires changing the configuration variable [```labels_class_boundaries ```](https://github.com/urban-resilience-lab/deepcovidnet/blob/eea555af4b5711e12e0c607a7f118bdfc38a22e8/deepcovidnet/config/CovidCountyDatasetConfig.py#L12) to define new end points for the output class ranges.

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
