import sqlite3

from configs import BASE_DIR, DB_PATH

conn = sqlite3.connect(DB_PATH)

c = conn.cursor()

with open(BASE_DIR / 'scripts/yelp_db.sql', 'r') as f:
    c.executescript(f.read())

conn.commit()
conn.close()
