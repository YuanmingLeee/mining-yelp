import pickle

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from configs import DATA_DIR

with open(DATA_DIR / 'logistical-data-loaders.pkl', 'rb') as f:
    train_set, test_set = pickle.load(f)

test_samples = test_set['features']
test_labels = test_set['label']

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
                           missing=None,
                           )
print('Training')
xg_cla.fit(train_set['features'], train_set['label'].reshape((-1,)), verbose=True)

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
