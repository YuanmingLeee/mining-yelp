import sqlite3
import pandas as pd

from configs import BASE_DIR, DB_PATH


def csv_prepare():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("""
        SELECT u.user_id,
            review_count,
            useful,
            cool,
            funny,
            fans,
            (compliment_hot + compliment_more + compliment_profile + compliment_cute + compliment_list + compliment_note 
            + compliment_plain + compliment_cool + compliment_funny + compliment_writer + compliment_photos) 
            AS compliment,
            friends,
            tip_compliment,
            tips,
            review_num,
            review_len,
            u.user_id IN (SELECT user_id FROM user_elite) AS elite
        FROM users u,
             (SELECT user_id, COUNT(*) AS friends FROM user_friends GROUP BY user_id) uf,
             (SELECT user_id, SUM(compliment_count) AS tip_compliment, COUNT(*) as tips FROM tip GROUP BY user_id) t,
             (
                SELECT user_id, COUNT(*) AS review_num, AVG(LENGTH(text)) AS review_len
                FROM review
                WHERE STRFTIME('%Y', date) BETWEEN '2010' AND '2016'
                GROUP BY user_id
             ) r
        WHERE u.user_id = uf.user_id
        AND   u.user_id = t.user_id
        AND   u.user_id = r.user_id;
        """, conn)
        df.to_csv(BASE_DIR / 'output/user-profiling.csv', index=False)


if __name__ == '__main__':
    csv_prepare()
