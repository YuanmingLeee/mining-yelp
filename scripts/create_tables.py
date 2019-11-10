import sqlite3

from configs import BASE_DIR

conn = sqlite3.connect(BASE_DIR / 'data/yelp.sqlite')

c = conn.cursor()

with open(BASE_DIR / 'scripts/yelp_db.sql', 'r') as f:
    c.executescript(f.read())

conn.commit()
conn.close()
