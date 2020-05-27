import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import sys

baseFileName = "111111111111"
if len(sys.argv) > 1 :
  baseFileName = str(sys.argv[1])

fileResults = open("lr"+baseFileName + ".results", "w")

np.random.seed(0)
X, y = load_svmlight_file("dataset/" + baseFileName + ".svmlight")
X_df = pd.DataFrame(X.todense())
y_df = pd.DataFrame(y)
# NUMBER_OF_TRAINING_SET_SIZE = 178778 # v1
# NUMBER_OF_TRAINING_SET_SIZE = 178684 # v2
NUMBER_OF_TRAINING_SET_SIZE = 178641  # v3
X_df_train = X_df.head(NUMBER_OF_TRAINING_SET_SIZE)
y_df_train = y_df.head(NUMBER_OF_TRAINING_SET_SIZE)
msk = np.random.rand(len(X_df_train)) < 0.95  # 0.8
y_train = y_df_train[msk]
x_train = X_df_train[msk]
y_validation = y_df_train[~msk]
x_validation = X_df_train[~msk]
NUMBER_OF_TEST_SET_SIZE = X_df.shape[0] - NUMBER_OF_TRAINING_SET_SIZE
x_df_test = X_df.tail(NUMBER_OF_TEST_SET_SIZE)
y_df_test = y_df.tail(NUMBER_OF_TEST_SET_SIZE)
# NUMBER_OF_TEST_SET_SIZES = [37092] # v1
# NUMBER_OF_TEST_SET_SIZES = [18114, 19072] # v2
NUMBER_OF_TEST_SET_SIZES = [8953, 9204, 8473, 8094, 10854]  # v3
x_test = []
y_test = []
sum_of_test_sets_sizes = 0
i = 0
for number_of_test_set_size in NUMBER_OF_TEST_SET_SIZES:
  sum_of_test_sets_sizes += number_of_test_set_size
  x_test.append(x_df_test.head(sum_of_test_sets_sizes).tail(number_of_test_set_size))
  y_test.append(y_df_test.head(sum_of_test_sets_sizes).tail(number_of_test_set_size))
  print(str(x_df_test.shape),"head("+str(sum_of_test_sets_sizes)+").tail("+str(number_of_test_set_size)+")")
  #print(str(x_test[i].shape), str(y_test[i].shape))
  i += 1

model = LogisticRegression(C=1e20,solver="liblinear")
model.fit(x_train, y_train.values.ravel())
y_pred_lr_train2 = model.predict(x_train)

predictions_lr_train2 = y_pred_lr_train2
accuracy_lr_train2= accuracy_score(y_train, predictions_lr_train2)
f1_lr_train2= f1_score(y_train, predictions_lr_train2)
precision_lr_train2 = precision_score(y_train, predictions_lr_train2)
recall_lr_train2 = recall_score(y_train, predictions_lr_train2)
#fpr_lr_train2, tpr_lr_train, thresholds_lr_train = roc_curve(y_train, y_pred_lr_train2)
#auc_lr_train2 = auc(fpr_lr_train2, tpr_lr_train)
print('LogisticRegressionTrain',accuracy_lr_train2,f1_lr_train2,precision_lr_train2,recall_lr_train2,file=fileResults)

y_pred_lr_valid2 = model.predict(x_validation)
predictions_lr_valid2 = y_pred_lr_valid2
accuracy_lr_valid2= accuracy_score(y_validation, predictions_lr_valid2)
f1_lr_valid2= f1_score(y_validation, predictions_lr_valid2)
precision_lr_valid2 = precision_score(y_validation, predictions_lr_valid2)
recall_lr_valid2 = recall_score(y_validation, predictions_lr_valid2)
#fpr_lr_valid2, tpr_lr_valid2, thresholds_lr_valid2 = roc_curve(y_validation, y_pred_lr_valid2)
#auc_lr_valid2 = auc(fpr_lr_valid2, tpr_lr_valid2)
print('LogisticRegressionValid',accuracy_lr_valid2,f1_lr_valid2,precision_lr_valid2,recall_lr_valid2,file=fileResults)
outstr = ""+str(accuracy_lr_valid2)+" "+str(f1_lr_valid2)+" "+str(precision_lr_valid2)+" "+str(recall_lr_valid2)
i = 1
for x_test_i, y_test_i in zip(x_test, y_test):
  predictions_lr2 = model.predict(x_test_i)
  accuracy_lr2 = accuracy_score(y_test_i, predictions_lr2)
  f1_lr2 = f1_score(y_test_i, predictions_lr2)
  precision_lr2 = precision_score(y_test_i, predictions_lr2)
  recall_lr2 = recall_score(y_test_i, predictions_lr2)
  # fpr_lr2, tpr_lr2, thresholds_lr2 = roc_curve(y_test_i, predictions_lr2)
  # auc_lr2 = auc(fpr_lr2, tpr_lr2)
  # print(str(x_test_i.shape), str(y_test_i.shape), file=fileResults)
  outstr = outstr + " " + str(accuracy_lr2) + " " + str(f1_lr2) + " " + str(precision_lr2) + " " + str(recall_lr2)
  print('LogisticRegressionTest' + str(i),accuracy_lr2,f1_lr2,precision_lr2,recall_lr2 ,file=fileResults)
  i += 1
print(outstr, file=fileResults)

fileResults.close()
