from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
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


rf = RandomForestClassifier(max_depth=3, n_estimators=10)
rf.fit(x_train, y_train.values.ravel())
y_pred_rf = rf.predict_proba(x_test)[:, 1]
predictions_rf = y_pred_rf
accuracy_rf= accuracy_score(y_test, predictions_rf)
f1_rf= f1_score(y_test, predictions_rf)
precision_rf = precision_score(y_test, predictions_rf)
print('RandomForest            |', accuracy_rf, '|', f1_rf, '|', precision_rf, '|', )
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
auc_rf = auc(fpr_rf, tpr_rf)


model_lr = LogisticRegression(C=1e20,solver="liblinear")
model_lr.fit(x_train, y_train.values.ravel())
y_pred_lr = model_lr.predict(x_test)
print('LogisticRegression      | ',accuracy_score(y_test, y_pred_lr),'     | ',f1_score(y_test, y_pred_lr),'      | ',precision_score(y_test, y_pred_lr))
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_lr)
auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_xgb, tpr_xgb, label='XGB (area = {:.25f})'.format(auc_xgb))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.25f})'.format(auc_rf))
plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.25f})'.format(auc_lr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('bl1.png')
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_xgb, tpr_xgb, label='XGB (area = {:.25f})'.format(auc_xgb))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.25f})'.format(auc_rf))
plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.25f})'.format(auc_lr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.savefig('bl2.png')