# DeepCOVIDNet: An Interpretable Deep Learning Model for Predictive Surveillance of COVID-19 Using Heterogeneous Features and Their Interactions

This is the official GitHub repository of the DeepCOVIDNet model to forecast the range of increase in the number of infected cases in all U.S. counties. For full details, refer to the original paper at https://arxiv.org/abs/2008.00115.

## Instructions to run

For full functionality, use Python 3.7+. If not available, Python 3.6 will work but will not support automatic hyperparameter tuning. 

```bash
#setup the project
python3 setup.py develop
#start working/testing now
```
Note that most interaction with this code will require changing file directories in the ```deepcovidnet/config/``` folder to match directories in the user's machine. Of particular importance is to change the values of variables ```data_base_dir``` to a location where all raw data will be stored and ```data_save_dir``` to a location where all saved data must be stored in the ```deepcovidnet/config/global_config.py``` file.

Note that the ```data_base_dir``` directory expects certain structure inside to be parseable easily. This structure can be changed by modifying the ```deepcovidnet/deepcovidnet/config/RawFeatureExtractorConfig.py``` file, but the default expected structure is outlined below. The ```data_base_dir``` is expected to contain the following subdirectories:
 - ```core_places/```: Location of [SafeGraph Core Places data](https://docs.safegraph.com/v4.0/docs#section-core-places)
 - ```safegraph_open_census_data/```: Location of [SafeGraph Open Census Data](https://docs.safegraph.com/docs/open-census-data)
 - ```social_distancing/```: Location of [SafeGraph Social Distancing Metrics](https://docs.safegraph.com/docs/social-distancing-metrics). The data for date ```YYYY-MM-DD``` should be in ```social_distancing/YYYY/MM/DD/YYYY-MM-DD-social-distancing.csv```
 - ```monthly_patterns/```: Location of [SafeGraph Monthly Patterns](https://docs.safegraph.com/v4.0/docs/places-schema#section-patterns). The data for month ```YYYY/MM``` should be in ```monthly_patterns/YYYYMM-AllPatterns-PATTERNS-YYYY_MM/patterns-part*.csv```
 - ```weekly_patterns/```: Location of [SafeGraph Weekly Patterns](https://docs.safegraph.com/v4.0/docs/places-schema#section-patterns). The data for the week starting on ```YYYY-MM-DD``` should be in ```weekly_patterns/YYYY-MM-DD-weekly-patterns.csv```
 
For more information, refer to ```deepcovidnet/deepcovidnet/config/RawFeatureExtractorConfig.py```. Depending upon the use case, some of the above directories can be unnecessary if the user does not plan to use the features in a particular subdirectory. To modify which features to include in the model, modify [```deepcovidnet/CovidCountyDataset.py```](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/CovidCountyDataset.py#L41).
 
Some feature used in the model like Venables Distance and reproduction number were developed internally in the lab and are not available in public. Feel free to open an issue if you are interested in getting the data to understand more about the data sharing policy of the lab.
 
 ### Adding Features
 
 Including additional features is easy. To add a new feature, simply define a function in ```deepcovidnet/RawFeatureExtractor.py``` that parses raw data, a function in ```deepcovidnet/DataSaver.py``` to save the processed feature value from the raw data, and a function in ```deepcovidnet/DataLoader.py``` to load the saved feature. Refer the example of the following three functions to understand more: [```read_sg_social_distancing```](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/RawFeatureExtractor.py#L287), [```save_sg_social_distancing```](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/DataSaver.py#L29), and [```load_sg_social_distancing```](https://github.com/urban-resilience-lab/deepcovidnet/blob/master/deepcovidnet/DataLoader.py#L34).
 
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
