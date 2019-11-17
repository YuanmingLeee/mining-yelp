import time

import matplotlib.pyplot as plt
from gensim.models import Doc2Vec
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

from configs import DATA_DIR
from data_engine.data_loader import load_statistical_learning_data

start = time.time()

# load pretrained model
model_dbow = Doc2Vec.load(DATA_DIR / 'doc2vec.model')

# load dataset
train_set, test_set = load_statistical_learning_data(DATA_DIR / 'tagged-dataset.pkl', model_dbow)


svclassifier = SVC(kernel='rbf', C=10, gamma=10, probability=True)

svclassifier.fit(train_set['features'], train_set['label'])

test_samples = test_set['features']
test_labels = test_set['label']

preds = svclassifier.predict(test_samples)

# get accuracy, f1 score and confusion matrix
print('Testing accuracy %s' % accuracy_score(test_labels, preds))
print('Testing F1 score: {}'.format(f1_score(test_labels, preds, average='weighted')))
print('Testing Confusion Matrix score: {}'.format(confusion_matrix(test_labels, preds)))

# get ROC graph
y_pred_proba = svclassifier.predict_proba(test_samples)[::, 1]
fpr, tpr, _ = metrics.roc_curve(test_labels, y_pred_proba)
auc = metrics.roc_auc_score(test_labels, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

# get running time
end = time.time()
duration = end - start
print("duration = " + str(duration) + "s")
