import pickle

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from configs import DATA_DIR

with open(DATA_DIR / 'logistical-data-loaders.pkl', 'rb') as f:
    train_set, test_set = pickle.load(f)

# training
print('Training')
logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(train_set['features'], train_set['label'])

test_samples = test_set['features']
test_labels = test_set['label']

preds = logreg.predict(test_samples)

# get accuracy, f1 score and confusion matrix
print('Testing accuracy %s' % accuracy_score(test_labels, preds))
print('Testing F1 score: {}'.format(f1_score(test_labels, preds, average='weighted')))
print('Testing Confusion Matrix score: {}'.format(confusion_matrix(test_labels, preds)))

# get ROC graph
y_pred_proba = logreg.predict_proba(test_samples)[::, 1]
fpr, tpr, _ = metrics.roc_curve(test_labels, y_pred_proba)
auc = metrics.roc_auc_score(test_labels, y_pred_proba)

plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()
