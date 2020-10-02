import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from joblib import dump, load
import sys
import gc
import argparse
import multiprocessing
import time

#nohup python3 randomforest_p2p.py --core 50 --basename 1010rr1101rr10 --prefix 10c --test_since_set 10 --pred_file_name 1010rr1101rr10network100000xallHour2 --random_forest 10c-rf85-1010rr1101rr10.joblib  &> 10c-rf85-1010rr1101rr10.out &

parser=argparse.ArgumentParser()
parser.add_argument('--core', default=1, help='Number of core that used for parallel processes')
parser.add_argument('--basename', default="1010rr1101rr10", help='base name of dataset')
parser.add_argument('--prefix', default="", help='prefix for output filename')
parser.add_argument('--test_since_set', default=False, help='start the test from the specified test set with a number')
parser.add_argument('--pred_file_name', default=False, help='make prediction for the given file')
parser.add_argument('--random_forest', default=False, help='open random forest from file')
parser.add_argument('--number_of_pred_file', default=0, help='open random forest from file')


args=parser.parse_args()

NUMBER_OF_CORES = int(args.core)
baseFileName = str(args.basename)
prefix = str(args.prefix)
if str(args.test_since_set) != "False" and str(args.pred_file_name) != "false":
  isTestV4 = True
  testSinceSet = int(args.test_since_set) - 1
else :
  isTestV4 = False
  testSinceSet = 0
if str(args.pred_file_name) != "False" and str(args.pred_file_name) != "false":
  isCreatePredictionForSimulator = True
  predictionSamplesDFileBaseName = str(args.pred_file_name)
  predictionSamplesDFileBaseName_suffix =  predictionSamplesDFileBaseName.split('x')[1]
else:
  isCreatePredictionForSimulator = False
if str(args.random_forest) != "False" and str(args.random_forest) != "false" :
  isRandomForestInputFile = True
  randomForestInputFile = str(args.random_forest)
else:
  isRandomForestInputFile = False
NUMBER_OF_PRED_FILE = int(args.number_of_pred_file)

fileResults = open("rf"+prefix+""+baseFileName + ".results", "w")

np.random.seed(0)
X, y = load_svmlight_file("dataset/" + baseFileName + ".svmlight")
X_df = pd.DataFrame(X.todense())
y_df = pd.DataFrame(y)
print(str(X_df.shape), str(y_df.shape), file=fileResults)
# NUMBER_OF_TRAINING_SET_SIZE = 178778 # v1
# NUMBER_OF_TRAINING_SET_SIZE = 178684 # v2
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
y_validation_test = y_df_train[~msk]
x_validation_test = X_df_train[~msk]
next_msk = np.random.rand(len(x_validation_test)) < 0.6
y_vtest = y_validation_test[next_msk]
x_vtest = x_validation_test[next_msk]
y_validation = y_validation_test[~next_msk]
x_validation = x_validation_test[~next_msk]

print(str(NUMBER_OF_TRAINING_SET_SIZE), str(x_train.shape), str(x_vtest.shape), str(x_validation.shape), file=fileResults)
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

gc.collect()
max_depth = 85
if isRandomForestInputFile == True and randomForestInputFile != "" :
    print("rf2_load_start")
    rf2 = load(randomForestInputFile)
    print("rf2_load_end")
else :
    parameter_max_depth = {
        "1010rr1101rr10": 85,
        "0000rr1101rr10": 85,
        "0000rr1101r010": 85,
        "0000001101r010": 70
    }
    if baseFileName in parameter_max_depth :
        max_depth = parameter_max_depth.get(baseFileName)
    #rf2 = RandomForestClassifier(max_depth=200, n_estimators=2000,max_features=0.3)
    #rf2 = RandomForestClassifier(max_depth=85, n_estimators=2000,max_features=0.3)
    rf2 = RandomForestClassifier(max_depth=max_depth, n_estimators=1500,max_features="log2")
    rf2.fit(x_train, y_train.values.ravel())
    dump(rf2, 'rf' + str(max_depth) + "-" + prefix + baseFileName + '.joblib')



'''
y_pred_rf_train2 = rf2.predict(x_train)
predictions_rf_train2 = y_pred_rf_train2
accuracy_rf_train2= accuracy_score(y_train, predictions_rf_train2)
f1_rf_train2= f1_score(y_train, predictions_rf_train2)
precision_rf_train2 = precision_score(y_train, predictions_rf_train2)
recall_rf_train2 = recall_score(y_train, predictions_rf_train2)
#fpr_rf_train2, tpr_rf_train, thresholds_rf_train = roc_curve(y_train, y_pred_rf_train2)
#auc_rf_train2 = auc(fpr_rf_train2, tpr_rf_train)
print('RandomForestTrain2',accuracy_rf_train2,f1_rf_train2,precision_rf_train2,recall_rf_train2,file=fileResults)
'''
#print("gc.collect()_start")
gc.collect()
#print("gc.collect()_end")


def predict_samples(fileSuffixList, core) :
  print(core,len(fileSuffixList),str(fileSuffixList))
  fileToPrintZ = open(predictionSamplesDFileBaseName + "rf" + str(max_depth) + "-" + prefix +"-"+ str(core) + 'prediction.out', "w",
                      encoding="utf-8")
  for suffix in fileSuffixList:
    #print("load_svmlight_file_start" + suffix)
    Z_File_1, z_y_1 = load_svmlight_file(
      "../createP2PConnectionChurnModel/partOfSvmLight_"+predictionSamplesDFileBaseName_suffix+"/" + predictionSamplesDFileBaseName + "_" + suffix + ".svmlight")
    #print("load_svmlight_file_end" + suffix, type(Z_File_1))
    gc.collect()
    #print("svmlight_file_todense_start" + suffix)
    Z1_test_svmlight = pd.DataFrame(Z_File_1.todense())
    #print("svmlight_file_todense_end" + suffix + " " + str(Z1_test_svmlight.shape[1])+" "+str(X_df_train.shape[1]))
    #print(core,suffix,"add_new_empty_columns_to_Z1_test_svmlight_start")
    for i in range(Z1_test_svmlight.shape[1], X_df_train.shape[1]):
      Z1_test_svmlight['newCol' + str(i)] = Z1_test_svmlight.apply(lambda x: 0, axis=1)
    #print(core,suffix,"add_new_empty_columns_to_Z1_test_svmlight_end",str(Z1_test_svmlight.shape))
    del Z_File_1
    del z_y_1
    gc.collect()
    #print(core,suffix,"predict_start")
    Z1_pred_rf = rf2.predict(Z1_test_svmlight)
    print(core,suffix,"predict_end",str(Z1_test_svmlight.shape),str(Z1_pred_rf.shape))
    for prediction in Z1_pred_rf:
      fileToPrintZ.write('' + str(int(round(prediction))) + '\n')
    del Z1_test_svmlight
    del Z1_pred_rf
    gc.collect()
  fileToPrintZ.close()


if isCreatePredictionForSimulator == True:
  #MAIN
  fileList = {}
  for core in range(NUMBER_OF_CORES) :
    fileList[core] = []

  THREAD_FILE_NUMBER_BLOCK_SIZE = int(NUMBER_OF_PRED_FILE/NUMBER_OF_CORES)
  fi=0
  core = 0
  #print(str(0),sumOfSize,str(THREAD_FILE_SIZE_BLOCK_SIZE),str(NUMBER_OF_CORES*THREAD_FILE_SIZE_BLOCK_SIZE))
  for i in range(0,NUMBER_OF_PRED_FILE) :
    #searchObj = re.search(r'^[0-9]{6}_2014[0-9]{4}-[0-9]{4}\.csv$', fileName)
    fileList[core].append("p"+str(i))
    fi+=1
    if fi / THREAD_FILE_NUMBER_BLOCK_SIZE >= 1 and core != NUMBER_OF_CORES-1:
      core += 1
      #print(fi, sumOfSize, str(THREAD_FILE_SIZE_BLOCK_SIZE), str(NUMBER_OF_CORES-core))
      fi = 0
  #print(fi, sumOfSize, str(THREAD_FILE_SIZE_BLOCK_SIZE), str(NUMBER_OF_CORES-core))
  processes = []
  for core in range(NUMBER_OF_CORES) :
    processes.append(multiprocessing.Process(target=predict_samples, args=(fileList[core],core)))
    processes[-1].start()  # start the thread we just created
    time.sleep(30)
  for t in processes:
    t.join()





'''  
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
'''
fileResults.close()