import json
import sqlite3

import numpy as np
import pandas as pd

from configs import DATA_DIR

DB_PATH = DATA_DIR / 'yelp.sqlite'
BUSINESS_JSON_PATH = DATA_DIR / 'yelp_academic_dataset_business.json'
USER_JSON_PATH = DATA_DIR / 'yelp_academic_dataset_user.json'
REVIEW_JSON_PATH = DATA_DIR / 'yelp_academic_dataset_review.json'
CHECKIN_JSON_PATH = DATA_DIR / 'yelp_academic_dataset_checkin.json'
TIP_JSON_PATH = DATA_DIR / 'yelp_academic_dataset_tip.json'


def load_business():
    df: pd.DataFrame = pd.read_json(BUSINESS_JSON_PATH, lines=True)
    # conversion
    df.attributes = df.attributes.apply(json.dumps)
    df.hours = df.hours.apply(json.dumps)
    df.categories = df.categories.apply(json.dumps)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(name='business', con=conn, index=False, if_exists='append')
        conn.commit()


def load_user():
    df: pd.DataFrame = pd.read_json(USER_JSON_PATH, lines=True)
    # conversion
    df.yelping_since = pd.to_datetime(df.yelping_since)

    # free after commit
    with sqlite3.connect(DB_PATH) as conn:
        # print('Processing user data frame')
        # user_df = df.drop(columns=['friends', 'elite'])
        # print('Commit user data into database')
        # user_df.to_sql(name='users', con=conn, index=False, if_exists='append')
        # conn.commit()
        # del user_df
        #
        # print('Processing user friends data frame')
        # user_friends_df = list_to_multirow(df, 'user_id', 'friends')
        # user_friends_df.rename(columns={'friends': 'friend_user_id'}, inplace=True)
        # print('Commit user data into database')
        # user_friends_df.to_sql(name='user_friends', con=conn, index=False, if_exists='append')
        # conn.commit()
        # del user_friends_df

        print('Processing user elite data frame')
        elite_df = list_to_multirow(df, 'user_id', 'elite', separator=',')
        print('Commit user elite data into database')
        elite_df.to_sql(name='user_elite', con=conn, index=False, if_exists='append')
        conn.commit()
        del elite_df


def load_review():
    df: pd.DataFrame = pd.read_json(REVIEW_JSON_PATH, lines=True)
    # conversion
    print('Processing review data frame')
    df.date = pd.to_datetime(df.date)

    with sqlite3.connect(DB_PATH) as conn:
        print('Commit user review data into database')
        df.to_sql(name='review', con=conn, index=False, if_exists='append')
        conn.commit()


def load_checkin():
    df: pd.DataFrame = pd.read_json(CHECKIN_JSON_PATH, lines=True)
    with sqlite3.connect(DB_PATH) as conn:
        print('Commit checkin data into database')
        df.to_sql(name='checkin', con=conn, index=False, if_exists='append')
        conn.commit()


def load_tip():
    df: pd.DataFrame = pd.read_json(TIP_JSON_PATH, lines=True)
    print('Processing tip data frame')
    df.date = pd.to_datetime(df.date)
    with sqlite3.connect(DB_PATH) as conn:
        print('Commit tip data into database')
        df.to_sql(name='tip', con=conn, index=False, if_exists='append')
        conn.commit()


def list_to_multirow(df: pd.DataFrame, index_name: str, list_name: str, separator=', '):
    value_list = df[list_name].apply(lambda x: x.split(separator) if x != 'None' and x != '' else [])
    return pd.DataFrame({
        index_name: np.repeat(df[index_name].values, value_list.str.len()),
        list_name: np.concatenate(value_list)
    })


if __name__ == '__main__':
    print('Loading business')
    # load_business()
    print('Loading user')
    load_user()
    print('Loading review')
    # load_review()
    print('Loading checkin')
    # load_checkin()
    print('Loading tip')
    # load_tip()
