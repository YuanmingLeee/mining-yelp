import sqlite3

from configs import BASE_DIR, DATA_PATH

conn = sqlite3.connect(DATA_PATH / 'yelp.sqlite')

c = conn.cursor()

with open(BASE_DIR / 'scripts/yelp_db.sql', 'r') as f:
    c.executescript(f.read())

conn.commit()
conn.close()
