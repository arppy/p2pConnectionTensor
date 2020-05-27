import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectFromModel

#all_day_time
X, y = load_svmlight_file("dataset/101011110111.svmlight")
X_df = pd.DataFrame(X.todense())
y_df = pd.DataFrame(y)
np.random.seed(0)
msk = np.random.rand(len(X_df)) < 0.8
y_train = y_df[msk]
x_train = X_df[msk]
y_test = y_df[~msk]
x_test = X_df[~msk]

'''
train = df[msk]
test = df[~msk]
y_train = train.iloc[:, 0]
x_train = train.iloc[:, 1:]
y_test = test.iloc[:, 0]
x_test = test.iloc[:, 1:]
'''

# 101011110111 2048
rf = RandomForestClassifier(max_depth=100, n_estimators=1000)
embeded_rf_selector = SelectFromModel(rf, max_features=2048)

embeded_rf_selector.fit(X, y)

embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')
'''
rf.fit(x_train, y_train.values.ravel())
y_pred_rf = rf.predict(x_test)
predictions_rf = y_pred_rf
accuracy_rf = accuracy_score(y_test, predictions_rf)
f1_rf = f1_score(y_test, predictions_rf)
precision_rf = precision_score(y_test, predictions_rf)
print(' Feature Set            | Accuracy                         | F1 measure                       |  Precesion')
print('RandomForest            |', accuracy_rf, '|', f1_rf, '|', precision_rf, '|', )
'''