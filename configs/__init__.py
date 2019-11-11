from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent.parent
DATA_PATH = BASE_DIR / 'data'
DB_PATH = BASE_DIR / 'data/yelp.sqlite'
