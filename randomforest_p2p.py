import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from joblib import dump, load
import sys

baseFileName = "111111111111"
prefix = ""
isTestV4 = False
testSinceSet = 0
if len(sys.argv) > 1 :
  baseFileName = str(sys.argv[1])
if len(sys.argv) > 2 :
  prefix = str(sys.argv[2])
if len(sys.argv) > 3:
    isTestV4 = True
    testSinceSet = int(sys.argv[3]) - 1

fileResults = open("rf"+prefix+""+baseFileName + ".results", "w")

np.random.seed(0)
X, y = load_svmlight_file("dataset/" + baseFileName + ".svmlight")
X_df = pd.DataFrame(X.todense())
y_df = pd.DataFrame(y)

# NUMBER_OF_TRAINING_SET_SIZE = 178778 # v1
# NUMBER_OF_TRAINING_SET_SIZE = 178684 # v2
NUMBER_OF_TRAINING_SET_SIZE = 178641  # v3
# NUMBER_OF_TEST_SET_SIZES = [37092] # v1
# NUMBER_OF_TEST_SET_SIZES = [18114, 19072] # v2
NUMBER_OF_TEST_SET_SIZES = [8953, 9204, 8473, 8094, 10854]  # v3
if isTestV4 == True:
    if testSinceSet > len(NUMBER_OF_TEST_SET_SIZES) - 1:
        testSinceSet = len(NUMBER_OF_TEST_SET_SIZES) - 1
    sum = 0
    i = 0
    for size in NUMBER_OF_TEST_SET_SIZES:
        if i == testSinceSet:
            break
        sum += size
        i += 1
    NUMBER_OF_TRAINING_SET_SIZE += sum
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

x_test = []
y_test = []
sum_of_test_sets_sizes = 0
i = 0
for number_of_test_set_size in NUMBER_OF_TEST_SET_SIZES:
    if isTestV4 == True and i < testSinceSet:
        i += 1
        continue
    sum_of_test_sets_sizes += number_of_test_set_size
    x_test.append(x_df_test.head(sum_of_test_sets_sizes).tail(number_of_test_set_size))
    y_test.append(y_df_test.head(sum_of_test_sets_sizes).tail(number_of_test_set_size))
    #print(str(x_test[i].shape), str(y_test[i].shape))


parameter_max_depth = {
    "1010rr1101rr10": 85,
    "0000rr1101rr10": 85,
    "0000rr1101r010": 85,
    "0000001101r010": 70
}
if baseFileName in parameter_max_depth :
    max_depth = parameter_max_depth.get(baseFileName)
else :
    max_depth = 85
#rf = RandomForestClassifier(max_depth=200, n_estimators=2000,max_features=0.3)
#rf2 = RandomForestClassifier(max_depth=85, n_estimators=2000,max_features=0.3)
rf2 = RandomForestClassifier(max_depth=max_depth, n_estimators=2000,max_features="log2")
rf2.fit(x_train, y_train.values.ravel())

y_pred_rf_train2 = rf2.predict(x_train)
predictions_rf_train2 = y_pred_rf_train2
accuracy_rf_train2= accuracy_score(y_train, predictions_rf_train2)
f1_rf_train2= f1_score(y_train, predictions_rf_train2)
precision_rf_train2 = precision_score(y_train, predictions_rf_train2)
recall_rf_train2 = recall_score(y_train, predictions_rf_train2)
#fpr_rf_train2, tpr_rf_train, thresholds_rf_train = roc_curve(y_train, y_pred_rf_train2)
#auc_rf_train2 = auc(fpr_rf_train2, tpr_rf_train)
print('RandomForestTrain2',accuracy_rf_train2,f1_rf_train2,precision_rf_train2,recall_rf_train2,file=fileResults)

y_pred_rf_valid2 = rf2.predict(x_validation)
predictions_rf_valid2 = y_pred_rf_valid2
accuracy_rf_valid2= accuracy_score(y_validation, predictions_rf_valid2)
f1_rf_valid2= f1_score(y_validation, predictions_rf_valid2)
precision_rf_valid2 = precision_score(y_validation, predictions_rf_valid2)
recall_rf_valid2 = recall_score(y_validation, predictions_rf_valid2)
#fpr_rf_valid2, tpr_rf_valid2, thresholds_rf_valid2 = roc_curve(y_validation, y_pred_rf_valid2)
#auc_rf_valid2 = auc(fpr_rf_valid2, tpr_rf_valid2)
print('RandomForestValid2',accuracy_rf_valid2,f1_rf_valid2,precision_rf_valid2,recall_rf_valid2,file=fileResults)
outstr = ""+str(accuracy_rf_valid2)+" "+str(f1_rf_valid2)+" "+str(precision_rf_valid2)+" "+str(recall_rf_valid2)
i = 1
for x_test_i, y_test_i in zip(x_test, y_test):
  predictions_rf2 = rf2.predict(x_test_i)
  accuracy_rf2 = accuracy_score(y_test_i, predictions_rf2)
  f1_rf2 = f1_score(y_test_i, predictions_rf2)
  precision_rf2 = precision_score(y_test_i, predictions_rf2)
  recall_rf2 = recall_score(y_test_i, predictions_rf2)
  # fpr_rf2, tpr_rf2, thresholds_rf2 = roc_curve(y_test_i, predictions_rf2)
  # auc_rf2 = auc(fpr_rf2, tpr_rf2)
  # print(str(x_test_i.shape), str(y_test_i.shape), file=fileResults)
  outstr = outstr + " " + str(accuracy_rf2) + " " + str(f1_rf2) + " " + str(precision_rf2) + " " + str(recall_rf2)
  print('RandomForestTest' + str(i),accuracy_rf2,f1_rf2,precision_rf2,recall_rf2 ,file=fileResults)
  i += 1
print(outstr, file=fileResults)
fileResults.close()
dump(rf2, 'rf'+prefix+""+baseFileName+'.joblib')
