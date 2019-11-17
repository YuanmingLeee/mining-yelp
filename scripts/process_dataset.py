import pickle
import sqlite3

import nltk
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm, trange

from configs import DB_PATH, DATA_DIR
from scripts.text_lstm_utils import review_preprocessing


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


def text_lstm():
    merged_review_csv_dir = DATA_DIR / "merged_data.csv"
    glove_embedding_dir = DATA_DIR / "glove.6B.50d.txt"
    output_dir = DATA_DIR
    review_preprocessing(merged_review_csv_dir, glove_embedding_dir, output_dir)


def build_doc2vec_model():
    """
    Build for the pretrained Doc2Vec model from processed dataset `data/merged_data.csv`. The training and test set
        are split with a ratio of 0.7. The tagged dataset and pretrained doc2vec model are saved as
        `data/tagged-dataset.pkl` and `doc2vec.model` respectively.
    """

    merged_review_csv_dir = DATA_DIR / 'merged_data.csv'

    df = pd.read_csv(merged_review_csv_dir, names=['text', 'label'], dtype={'text': str, 'label': str})

    train, test = train_test_split(df, test_size=0.3, random_state=42)

    def tokenize_text(text):
        """
        Tokenize review content used for logistic regression, SVM and XGBoost.

        Args:
            text: Text to be tokenized.
        Return:
            Tokens
        """
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    print('Processing dataset')
    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r.text), tags=r.label), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r.text), tags=r.label), axis=1)

    # build vocabulary
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=4)
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    print('Finish processing')

    # build pretained model
    print('Building pretrained model')
    for _ in trange(30):
        model_dbow.train(shuffle([x for x in train_tagged.values]),
                         total_examples=len(train_tagged.values),
                         epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha
    print('Finish building')

    # save dataset and pretained doc2vec model
    print('saving')
    with open(DATA_DIR / 'tagged-dataset.pkl', 'wb') as f:
        pickle.dump((train_tagged, test_tagged), f)

    model_dbow.save('doc2vec.model')


if __name__ == '__main__':
    user_elite_cleaned_csv()
    multimodal_classifier()
    text_lstm()
    build_doc2vec_model()
