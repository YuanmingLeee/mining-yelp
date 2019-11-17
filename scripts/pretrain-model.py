import pickle

import nltk
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm, trange

from configs import DATA_DIR
from data_engine.data_loader import load_statistical_learning_data
from scripts.text_lstm_utils import review_preprocessing


def text_lstm():
    """
    Build Text LSTM mapping and pretrained weights for embedding
    """
    merged_review_csv_dir = DATA_DIR / "merged_data.csv"
    glove_embedding_dir = DATA_DIR / "glove.6B.50d.txt"
    output_dir = DATA_DIR
    review_preprocessing(merged_review_csv_dir, glove_embedding_dir, output_dir)


def build_doc2vec_model():
    """
    Build for the pretrained Doc2Vec model from processed dataset `data/merged_data.csv`. The training and test set
        are split with a ratio of 0.7. The tagged dataset and pretrained doc2vec model are saved as
        `data/tagged-dataset.pkl` and `doc2vec.model{,trainables.syn1neg.npy,wv.vectors.npy}` respectively.
        This function also saves a processed data loader for boost up.
    """

    merged_review_csv_dir = DATA_DIR / 'c.csv'

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

    # enable pandas tqdm
    tqdm.pandas()

    print('Processing dataset')
    print('Tagging train set')
    train_tagged = train.progress_apply(
        lambda r: TaggedDocument(words=tokenize_text(r.text), tags=r.label), axis=1)
    print('Tagging test set')
    test_tagged = test.progress_apply(
        lambda r: TaggedDocument(words=tokenize_text(r.text), tags=r.label), axis=1)

    # build vocabulary
    print('Building vocabulary')
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
    print('Saving')
    with open(DATA_DIR / 'tagged-dataset.pkl', 'wb') as f:
        pickle.dump((train_tagged, test_tagged), f)

    model_dbow.save(str(DATA_DIR / 'doc2vec.model'))

    print('Building data loader for speed up')
    train_set, test_set = load_statistical_learning_data(DATA_DIR / 'tagged-dataset.pkl', model_dbow)

    with open(DATA_DIR / 'statistical-data-loaders.pkl', 'wb') as f:
        pickle.dump((train_set, test_set), f)

    print('Finish')


if __name__ == '__main__':
    print('Text LSTM...')
    # text_lstm()
    print('Doc2Vec...')
    build_doc2vec_model()
