import time

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import metrics
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from configs import DATA_DIR

start = time.time()


def tokenize_text(text):
    """Tokenize review content"""
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def train_vocab(train, test):
    """
    train vocabulary and train model
    """
    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r.text), tags=r.label), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r.text), tags=r.label), axis=1)

    # building vocabulary

    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=4)
    model_dbow.build_vocab(train_tagged.values)

    # Initialise model
    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in train_tagged.values]), total_examples=len(train_tagged.values),
                         epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    return train_tagged, test_tagged, model_dbow


def vec_for_learning(model, tagged_docs):
    """get doc vector"""
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


CSV_PATH = DATA_DIR / 'merged_data.csv'

df = pd.read_csv(CSV_PATH, names=['text', 'label'], dtype={'text': str, 'label': str})

train, test = train_test_split(df, test_size=0.3, random_state=42)

train_tagged, test_tagged, model_dbow = train_vocab(train, test)

y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
y_test = [int(item) for item in y_test]
y_pred = [int(item) for item in y_pred]

"""get accuracy, f1 score and confusion matrix"""
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Testing Confusion Matrix score: {}'.format(confusion_matrix(y_test, y_pred)))

"""get ROC graph"""
y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

"""get running time"""
end = time.time()
duration = end - start
print("duration = " + str(duration) + "s")
