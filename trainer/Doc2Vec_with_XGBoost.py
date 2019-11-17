import nltk
import pandas as pd
import pymongo
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import TaggedDocument
import xgboost as xgb
from sklearn import metrics
import time

start = time.time()
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def DBconnect(host, db, col):
    """connect to mondoDB"""
    client = pymongo.MongoClient(host)
    myDb = client[db]
    myCollection = myDb[col]
    return myCollection


def getDF(Col, size):
    """get data in dataframe format"""
    DF0 = pd.DataFrame(list(Col.find({"label": "0"}).limit(size)))
    label = [str(0) for row in DF0.itertuples()]
    DF0["label"] = label

    DF1 = pd.DataFrame(list(Col.find({"$or": [{"label": "1"}, {"label": 1}]}).limit(size)))
    label = [str(1) for row in DF1.itertuples()]
    DF1["label"] = label
    ReviewDF = pd.concat([DF0, DF1])
    ReviewDF = ReviewDF[['text', 'label']]

    return ReviewDF


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
    """train vocabulary and train model"""
    train_tagged = train.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=r.label), axis=1)
    test_tagged = test.apply(
        lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=r.label), axis=1)

    """Building vocabulary"""

    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=4)
    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

    """Initialise model"""

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                         epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    return train_tagged, test_tagged, model_dbow


def vec_for_learning(model, tagged_docs):
    """get doc vector"""
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


ReviewsCol = DBconnect("mongodb://localhost:27017/", "Yelp", "ProjectDA")
ReviewDF = getDF(ReviewsCol, 100)
train, test = train_test_split(ReviewDF, test_size=0.3, random_state=42)

train_tagged, test_tagged, model_dbow = train_vocab(train, test)

X_train = vec_for_learning(model_dbow, train_tagged)
X_train_DF = pd.DataFrame(X_train)

X_test = vec_for_learning(model_dbow, test_tagged)
X_test_DF = pd.DataFrame(X_test)

y_train = train["label"]
y_test = test["label"]

X_test_DM = xgb.DMatrix(X_test_DF)

xg_cla = xgb.XGBClassifier(max_depth=3,
                           min_child_weight=1,
                           learning_rate=0.1,
                           n_estimators=500,
                           silent=True,
                           objective='binary:logistic',
                           gamma=0,
                           max_delta_step=0,
                           subsample=1,
                           colsample_bytree=1,
                           colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=1,
                           missing=None)
xg_cla.fit(X_train_DF, y_train)

y_pred = xg_cla.predict(X_test_DF)

y_test = [int(item) for item in y_test]
y_pred = [int(item) for item in y_pred]

"""get accuracy, f1 score and confusion matrix"""
print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
print('Testing Confusion Matrix score: {}'.format(confusion_matrix(y_test, y_pred)))

"""get ROC graph"""
y_pred_proba = xg_cla.predict_proba(X_test_DF)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

"""get running time"""
end = time.time()
duration = end - start
print("duration = " + str(duration) + "s")
