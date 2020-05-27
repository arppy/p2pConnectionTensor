from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X, y = load_svmlight_file("dataset/all_day_time.svmlight")
X_df = pd.DataFrame(X.todense())
y_df = pd.DataFrame(y)
np.random.seed(0)
msk = np.random.rand(len(X_df)) < 0.8
y_train = y_df[msk]
x_train = X_df[msk]
y_test = y_df[~msk]
x_test = X_df[~msk]

model = XGBClassifier()
model.fit(x_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(x_test)
predictions = y_pred
#predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy= accuracy_score(y_test, predictions)
f1= f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
print('Feature Set            | Accuracy                         | F1 measure                       |  Precesion')
print('XGBoost                |', accuracy, '|', f1, '|', precision, '|', )
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_pred)
auc_xgb = auc(fpr_xgb, tpr_xgb)

fig, ax = plt.subplots(figsize=(30, 30))
plot_tree(model, num_trees=4, ax=ax)
plt.savefig("tree.pdf")