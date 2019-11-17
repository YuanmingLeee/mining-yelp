import time

import matplotlib.pyplot as plt
import xgboost as xgb
from gensim.models import Doc2Vec
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from configs import DATA_DIR
from data_engine.data_loader import load_statistical_learning_data

start = time.time()

# load pretrained model
model_dbow = Doc2Vec.load(DATA_DIR / 'doc2vec.model')

# load dataset
train_set, test_set = load_statistical_learning_data(DATA_DIR / 'tagged-dataset.pkl', model_dbow)

test_samples = test_set['features']
test_labels = test_set['label']

# X_train = np.asarray(X_train)
# X_test = np.asarray(X_test)

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
xg_cla.fit(train_set['features'], train_set['label'])

# test_DM = xgb.DMatrix(test_samples, test_labels)
preds = xg_cla.predict(test_samples)

# get accuracy, f1 score and confusion matrix"""
print('Testing accuracy %s' % accuracy_score(test_labels, preds))
print('Testing F1 score: {}'.format(f1_score(test_labels, preds, average='weighted')))
print('Testing Confusion Matrix score: {}'.format(confusion_matrix(test_labels, preds)))

# get ROC graph"""
y_pred_proba = xg_cla.predict_proba(test_samples)[::, 1]
fpr, tpr, _ = metrics.roc_curve(test_labels, y_pred_proba)
auc = metrics.roc_auc_score(test_labels, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

# get running time"""
end = time.time()
duration = end - start
print("duration = " + str(duration) + "s")
