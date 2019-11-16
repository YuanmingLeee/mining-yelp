from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
DB_PATH = BASE_DIR / 'data/yelp.sqlite'
