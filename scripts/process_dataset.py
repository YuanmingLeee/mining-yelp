import sqlite3

import pandas as pd

from configs import DB_PATH, DATA_DIR


def user_elite_csv():
    """
    Prepare data from database and save the queried table as a csv file.
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # pre-condition
        c.executescript("""
        -- create user friend summary table
        CREATE TEMP TABLE _uf AS
        SELECT user_id, COUNT(*) as friends
        FROM user_friends
        GROUP BY user_id;
        
        -- create user tip summary table
        CREATE TEMP TABLE _t AS
        SELECT user_id, SUM(compliment_count) AS tip_compliment, COUNT(*) AS tips, AVG(length(text)) AS tip_len
        FROM tip
        GROUP BY user_id;
        
        -- create user review summary table
        CREATE TEMP TABLE _r AS
        SELECT user_id, COUNT(*) as review_num, AVG(length(text)) as review_len
        FROM review
        GROUP BY user_id;
        
        -- add users having 0 friends
        INSERT INTO _uf
        SELECT user_id, 0
        FROM users
        WHERE user_id NOT IN (
            SELECT user_id
            FROM _uf
        );
        
        -- add users having 0 reviews
        INSERT INTO _r
        SELECT user_id, 0, 0.
        FROM users
        WHERE user_id NOT IN (
            SELECT user_id
            FROM _r
        );
        
        -- add users having 0 tips
        INSERT INTO _t
        SELECT user_id, 0, 0, 0.
        FROM users
        WHERE user_id NOT IN (
            SELECT user_id
            FROM _t
        );
        """)

        # noinspection SqlResolve
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
            tip_len,
            review_num,
            review_len,
            u.user_id IN (SELECT user_id FROM user_elite) AS elite
        FROM users u,
             _uf,
             _t,
             _r
        WHERE u.user_id = _uf.user_id
        AND   u.user_id = _t.user_id
        AND   u.user_id = _r.user_id;
        """, conn)
        df.to_csv(DATA_DIR / 'user-elite.csv', index=False)

        # post-condition
        # noinspection SqlResolve
        c.executescript("""
        DROP TABLE _t;
        DROP TABLE _uf;
        DROP TABLE _r;
        """)


def user_elite_cleaned_csv():
    """Prepare data from database and save the queried table as a csv file. This version
    only include reviews / tips years between 2010-2016. Elite users before 2010 or
    after 2016 are excluded.
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # pre-condition
        c.executescript("""
        -- create user friend summary table
        CREATE TEMP TABLE _uf AS
        SELECT user_id, COUNT(*) as friends
        FROM user_friends
        GROUP BY user_id;

        -- create user tip summary table
        CREATE TEMP TABLE _t AS
        SELECT user_id, SUM(compliment_count) AS tip_compliment, COUNT(*) AS tips, AVG(length(text)) AS tip_len
        FROM tip
        WHERE STRFTIME('%Y', date) BETWEEN '2010' AND '2016'
        GROUP BY user_id;

        -- create user review summary table
        CREATE TEMP TABLE _r AS
        SELECT user_id, COUNT(*) as review_num, AVG(length(text)) as review_len
        FROM review
        WHERE STRFTIME('%Y', date) BETWEEN '2010' AND '2016'
        GROUP BY user_id;

        -- add users having 0 friends
        INSERT INTO _uf
        SELECT user_id, 0
        FROM users
        WHERE user_id NOT IN (
            SELECT user_id
            FROM _uf
        );

        -- add users having 0 reviews
        INSERT INTO _r
        SELECT user_id, 0, 0.
        FROM users
        WHERE user_id NOT IN (
            SELECT user_id
            FROM _r
        );

        -- add users having 0 tips
        INSERT INTO _t
        SELECT user_id, 0, 0, 0.
        FROM users
        WHERE user_id NOT IN (
            SELECT user_id
            FROM _t
        );
        
        -- create exclusive user table
        CREATE TEMP TABLE _eu AS
        SELECT user_id
        FROM user_elite
        WHERE elite BETWEEN '2010' AND '2016'
        GROUP BY user_id;
        """)

        # noinspection SqlResolve
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
            tip_len,
            review_num,
            review_len,
            u.user_id IN (SELECT user_id FROM user_elite) AS elite
        FROM users u,
             _uf,
             _t,
             _r
        WHERE u.user_id = _uf.user_id
        AND   u.user_id = _t.user_id
        AND   u.user_id = _r.user_id
        AND   u.user_id NOT IN _eu
        """, conn)
        df.to_csv(DATA_DIR / 'user-elite-cleaned.csv', index=False)

        # post-condition
        # noinspection SqlResolve
        c.executescript("""
        DROP TABLE _t;
        DROP TABLE _uf;
        DROP TABLE _r;
        DROP TABLE _eu;
        """)


def multimodal_classifier():
    """
    Create dataset used for multimodal classifier
    """
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # create table from csv
        df: pd.DataFrame = pd.read_csv(DATA_DIR / 'user-profiling-cleaned.csv')
        df2: pd.DataFrame = pd.read_sql("""
            SELECT text, user_id, (useful > 10) as usefulness 
            FROM review 
            WHERE STRFTIME('%Y', date) BETWEEN '2010' AND '2016'
        """, con=conn)
        # noinspection SqlResolve
        df_join = pd.merge(df, df2, on='user_id')
        df_join.to_csv(DATA_DIR / 'combined-usefulness.csv', index=False)


if __name__ == '__main__':
    user_elite_csv()
    user_elite_cleaned_csv()
    multimodal_classifier()
