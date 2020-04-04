import os
from pathlib import Path
import getpass
from datetime import datetime
import multiprocessing

EPOCHS                      = 300

PRINT_FREQ                  = 20
INTERMITTENT_OUTPUT_FREQ    = 5 # Num batches between outputs

BATCH_SIZE              = 64

MIN_LEARNING_RATE           = 0.000001

TENSORBOARDX_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + getpass.getuser()))
MODELS_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'models')
DATA_BASE_DIR = os.path.join(
    Path(os.path.dirname(os.path.abspath(__file__))).parent, 'data')

### BaseCountyDataset
PLACE_COUNTY_CBG_FILE = os.path.join(DATA_BASE_DIR, 'placeCountyCBG.csv')
PATH_TO_SAFEGRAPH_OPEN_CENSUS_DATA = os.path.join(DATA_BASE_DIR, "safegraph_open_census_data/data/")

CORE_POI_PATH = os.path.join(DATA_BASE_DIR, 'CoreRecords-CORE_POI-2019_03-2020-03-25')
CORE_POI_CSV_PREFIX = 'core_poi-part'