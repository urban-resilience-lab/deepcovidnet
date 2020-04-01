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
