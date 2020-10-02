import pandas as pd
import numpy as np
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

baseFileName = "1010rr1101rr10"
prefix = ""
isTestV4 = False
testSinceSet = 0
if len(sys.argv) > 1:
  baseFileName = str(sys.argv[1])
if len(sys.argv) > 2:
  prefix = str(sys.argv[2])
if len(sys.argv) > 3:
  isTestV4 = True
  testSinceSet = int(sys.argv[3]) - 1

fileResults = open(prefix + "" + baseFileName + ".results", "w")

np.random.seed(0)
X_df = pd.read_csv("dataset/" + baseFileName + ".csv", header=None, skiprows=1)
y_df = X_df[0]
X_df = X_df.drop([0], axis=1)
# NUMBER_OF_TRAINING_SET_SIZE = 178778 # v1
# NUMBER_OF_TRAINING_SET_SIZE = 178684 # v2
#NUMBER_OF_TRAINING_SET_SIZE = 178641  # v3
NUMBER_OF_TRAINING_SET_SIZE = 187954  # v3
NUMBER_OF_TEST_SET_SIZES = [9352, 9792, 9224, 9230, 9233, 9283, 9915, 9916, 8587, 8883]  # v4
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
print(str(NUMBER_OF_TRAINING_SET_SIZE), str(x_train.shape), str(x_validation.shape), file=fileResults)
NUMBER_OF_TEST_SET_SIZE = X_df.shape[0] - NUMBER_OF_TRAINING_SET_SIZE
x_df_test = X_df.tail(NUMBER_OF_TEST_SET_SIZE)
y_df_test = y_df.tail(NUMBER_OF_TEST_SET_SIZE)
# NUMBER_OF_TEST_SET_SIZES = [37092] # v1
# NUMBER_OF_TEST_SET_SIZES = [18114, 19072] # v2

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
  # print(str(x_test[i].shape),str(y_test[i].shape))

print(i,testSinceSet,str(sys.argv[3]), file=fileResults)

print(y_train.mean(), file=fileResults)
y_train_mean = y_train.mean()
random_valid = pd.DataFrame(np.random.rand(len(y_validation)))
random_valid_prediction = pd.DataFrame(np.multiply((random_valid < y_train_mean),1))
accuracy_random_valid2= accuracy_score(y_validation, random_valid_prediction)
f1_random_valid2= f1_score(y_validation, random_valid_prediction)
precision_random_valid2 = precision_score(y_validation, random_valid_prediction)
recall_random_valid2 = recall_score(y_validation, random_valid_prediction)
print('RandomValid',accuracy_random_valid2,f1_random_valid2,precision_random_valid2,recall_random_valid2,file=fileResults)
outstr = ""+str(accuracy_random_valid2)+" "+str(f1_random_valid2)+" "+str(precision_random_valid2)+" "+str(recall_random_valid2)
y_pred_random = []
i = 1
for x_test_i, y_test_i in zip(x_test, y_test):
  random_vector = pd.DataFrame(np.random.rand(len(y_test_i)))
  predictions_random = pd.DataFrame(np.multiply((random_vector < y_train_mean), 1))
  y_pred_random.append(predictions_random)
  accuracy_random = accuracy_score(y_test_i, predictions_random)
  f1_random = f1_score(y_test_i, predictions_random)
  precision_random = precision_score(y_test_i, predictions_random)
  recall_random = recall_score(y_test_i, predictions_random)
  # fpr_rf2, tpr_rf2, thresholds_rf2 = roc_curve(y_test_i, predictions_rf2)
  # auc_rf2 = auc(fpr_rf2, tpr_rf2)
  # print(str(x_test_i.shape), str(y_test_i.shape), file=fileResults)
  outstr = outstr + " " + str(accuracy_random) + " " + str(f1_random) + " " + str(precision_random) + " " + str(recall_random)
  print('RandomTest' + str(i),accuracy_random,f1_random,precision_random,recall_random ,file=fileResults)
  i += 1
print(outstr, file=fileResults)
fileResults.close()
