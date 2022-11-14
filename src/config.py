from os.path import dirname, realpath
from pathlib import Path

ROOT_DIR = Path(dirname(realpath(__file__))).parent

# Data directory
DATA_DIR = ROOT_DIR / "data"

# Path to the raw data
RAW_DATA_PATH = DATA_DIR / "raw"

# Path to the processed data
PROCESSED_DATA_PATH = DATA_DIR / "processed"


SRC_DIR = ROOT_DIR / "src"
