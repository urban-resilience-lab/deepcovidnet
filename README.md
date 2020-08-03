# DeepCOVIDNet: An Interpretable Deep Learning Model for Predictive Surveillance of COVID-19 Using Heterogeneous Features and Their Interactions

This is the official GitHub repository of the DeepCOVIDNet model to forecast the range of increase in the number of infected cases in all U.S. counties.

## Instructions to run

```bash
#setup the project
python3 setup.py develop
#start working/testing now
```
Note that most interaction with this code will require changing file directories in the ```covid_county_prediction/config/``` folder to match directories in the user's machine. Of particular importance is to change the values of variables ```data_base_dir``` to a location where all raw data will be stored and ```data_save_dir``` to a location where all saved data must be stored in the ```covid_county_prediction/config/global_config.py/``` file.

Note that the ```data_base_dir``` directory expects certain structure inside to be parseable easily. This structure can be changed by modifying the ```covid-county-prediction/covid_county_prediction/config/RawFeatureExtractorConfig.py``` file, but the default expected structure is outlined below. The ```data_base_dir``` is expected to contain the following subdirectories:
 - ```core_places/```: Location of [SafeGraph Core Places data](https://docs.safegraph.com/v4.0/docs#section-core-places)
 - ```safegraph_open_census_data/```: Location of [SafeGraph Open Census Data](https://docs.safegraph.com/docs/open-census-data)
 - ```social_distancing/```: Location of [SafeGraph Social Distancing Metrics](https://docs.safegraph.com/docs/social-distancing-metrics). The data for date ```YYYY-MM-DD``` should be in ```social_distancing/YYYY/MM/DD/YYYY-MM-DD-social-distancing.csv```
 - ```monthly_patterns/```: Location of [SafeGraph Monthly Patterns](https://docs.safegraph.com/v4.0/docs/places-schema#section-patterns). The data for month ```YYYY/MM``` should be in ```monthly_patterns/YYYYMM-AllPatterns-PATTERNS-YYYY_MM/patterns-part*.csv```
 - ```weekly_patterns/```: Location of [SafeGraph Weekly Patterns](https://docs.safegraph.com/v4.0/docs/places-schema#section-patterns). The data for the week starting on ```YYYY-MM-DD``` should be in ```weekly_patterns/YYYY-MM-DD-weekly-patterns.csv```
 
For more information, refer to ```covid-county-prediction/covid_county_prediction/config/RawFeatureExtractorConfig.py```. Depending upon the use case, some of the above directories can be unnecessary if the user does not plan to use the features in a particular subdirectory. To modify which features to include in the model, modify [```covid_county_prediction/CovidCountyDataset.py```](https://github.com/urban-resilience-lab/covid-county-prediction/blob/master/covid_county_prediction/CovidCountyDataset.py#L41).
 
Some feature used in the model like Venables Distance and reproduction number were developed internally in the lab and are not available in public. Feel free to open an issue if you are interested in getting the data to understand more about the data sharing policy of the lab.
 
 ### Adding Features
 
 Including additional features is easy. To add a new feature, simply define a function in ```covid_county_prediction/RawFeatureExtractor.py``` that parses raw data, a function in ```covid_county_prediction/DataSaver.py``` to save the processed feature value from the raw data, and a function in ```covid_county_prediction/DataLoader.py``` to load the saved feature. Refer the example of the following three functions to understand more: [```read_sg_social_distancing```](https://github.com/urban-resilience-lab/covid-county-prediction/blob/master/covid_county_prediction/RawFeatureExtractor.py#L287), [```save_sg_social_distancing```](https://github.com/urban-resilience-lab/covid-county-prediction/blob/master/covid_county_prediction/DataSaver.py#L29), and [```load_sg_social_distancing```](https://github.com/urban-resilience-lab/covid-county-prediction/blob/master/covid_county_prediction/DataLoader.py#L34).
 
## Results

## Citation

Please cite this work if it was helpful in your research.
