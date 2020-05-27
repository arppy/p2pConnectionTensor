from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

baseFileName = "111111111111"
if len(sys.argv) > 1 :
  baseFileName = str(sys.argv[1])

X, y = load_svmlight_file("dataset/" + baseFileName + ".svmlight")
X_df = pd.DataFrame(X.todense())
y_df = pd.DataFrame(y)
np.random.seed(0)
msk = np.random.rand(len(X_df)) < 0.8
y_train = y_df[msk]
x_train = X_df[msk]
y_test = y_df[~msk]
x_test = X_df[~msk]

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train.values.ravel())
#Predict the response for test dataset
y_pred = clf.predict(x_test)
predictions = y_pred
#predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy= accuracy_score(y_test, predictions)
f1= f1_score(y_test, predictions)
precision = precision_score(y_test, predictions)
print(baseFileName)
print('Feature Set            | Accuracy                         | F1 measure                       |  Precesion')
print('DecisionTree           |', accuracy, '|', f1, '|', precision, '|', )
#fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_pred)
#auc_xgb = auc(fpr_xgb, tpr_xgb)

#dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True)
#graph = pydotplus.graph_from_dot_data(dot_data)
export_graphviz(clf, out_file=baseFileName+'.dot')
#graph.write_pdf('tree.pdf')