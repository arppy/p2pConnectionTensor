from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from joblib import dump, load

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import gc
import argparse

import matplotlib.pyplot as plt

import time
import os


def precision(y_true, y_prediction):
  """Precision metric.
  Only computes a batch-wise average of precision.
  Computes the precision, a metric for multi-label classification of
  how many selected items are relevant.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_prediction, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_prediction, 0, 1)))
  ret_precision = true_positives / (predicted_positives + K.epsilon())
  return ret_precision


def recall(y_true, y_prediction):
  """Recall metric.
  Only computes a batch-wise average of recall.
  Computes the recall, a metric for multi-label classification of
  how many relevant items are selected.
  """
  true_positives = K.sum(K.round(K.clip(y_true * y_prediction, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  ret_recall = true_positives / (possible_positives + K.epsilon())
  return ret_recall


def f1(y_true, y_prediction):
  ret_precision = precision(y_true, y_prediction)
  ret_recall = recall(y_true, y_prediction)
  return 2 * ((ret_precision * ret_recall) / (ret_precision + ret_recall + K.epsilon()))


def dropout_str(first_dropout, *dropouts):
  ret_dropout_str = "0:" + str(first_dropout) if first_dropout > 0 else ""
  i_dropout = 1
  for dropout in dropouts:
    if ret_dropout_str != "":
      ret_dropout_str = ret_dropout_str + "," + str(i_dropout) + ":" + str(
        dropout) if dropout > 0 else ret_dropout_str + ""
    else:
      ret_dropout_str = str(i_dropout) + ":" + str(dropout) if dropout > 0 else ""
    i_dropout += 1
  return ret_dropout_str


def layer_str(embeding_output, *layers):
  ret_layer_str = "E" + str(embeding_output)
  for layer in layers:
    ret_layer_str = ret_layer_str + "-D" + str(layer)
  return ret_layer_str


##
##
###
###
##

randomForestInputFile = ""
tensorflow2InputFile = ""
tensorflow1InputFile = ""

parser=argparse.ArgumentParser()
parser.add_argument('--core', default=1, help='the core that used for tensorflow')
parser.add_argument('--basename', default="1010rr1101rr10", help='base name of dataset')
parser.add_argument('--prefix', default="", help='prefix for output filename')
parser.add_argument('--test_since_set', default=False, help='start the test from the specified test set with a number')
parser.add_argument('--pred_file_name', default=False, help='make prediction for the given file suffix')
parser.add_argument('--pred_suffix', default="", help='make prediction for the given suffix')
parser.add_argument('--number_of_pred_file', default=0, help='number of prediction files')
parser.add_argument('--tensor_flow_input_file1', default=False, help='open deep network from file1')
parser.add_argument('--tensor_flow_input_file2', default=False, help='open deep network from file2')

args=parser.parse_args()

core = str(args.core)
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
predSuffix = str(args.pred_suffix)
NUMBER_OF_PRED_FILE = int(args.number_of_pred_file)
if str(args.tensor_flow_input_file1) != "False" and str(args.tensor_flow_input_file1) != "false" :
  isTensorflow1InputFile = True
  tensorflow1InputFile = str(args.tensor_flow_input_file1)
else:
  isTensorflow1InputFile = False
if str(args.tensor_flow_input_file2) != "False" and str(args.tensor_flow_input_file2) != "false":
  isTensorflow2InputFile = True
  tensorflow2InputFile = str(args.tensor_flow_input_file2)
else:
  isTensorflow2InputFile = False

print("PRINT","core",core,"baseFileName",baseFileName,"prefix",prefix,"isTestV4",isTestV4,"testSinceSet",testSinceSet,
      "isCreatePredictionForSimulator",isCreatePredictionForSimulator,"predictionSamplesDFileBaseName",predictionSamplesDFileBaseName,
      "predictionSamplesDFileBaseName_suffix",predictionSamplesDFileBaseName_suffix,"predSuffix",predSuffix,
      "NUMBER_OF_PRED_FILE",NUMBER_OF_PRED_FILE,"isTensorflow1InputFile",isTensorflow1InputFile,
      "isTensorflow1InputFile", isTensorflow1InputFile, "isTensorflow2InputFile", isTensorflow2InputFile,
      "tensorflow2InputFile",tensorflow2InputFile)
quit()

os.environ["CUDA_VISIBLE_DEVICES"] = core
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

fileResults = open(prefix + "" + baseFileName + ".results", "w")

tf.reset_default_graph()
tf.set_random_seed(0)
np.random.seed(0)
gpu_options = tf.compat.v1.GPUOptions(allocator_type="BFC", visible_device_list="0")
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, allow_soft_placement=True, log_device_placement=True, inter_op_parallelism_threads=1, gpu_options=gpu_options, device_count={'GPU': 1})
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, allow_soft_placement=True, log_device_placement=True,
                                  inter_op_parallelism_threads=0, gpu_options=gpu_options, device_count={'GPU': 4})
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
with sess.as_default():
  tf.set_random_seed(0)
  np.random.seed(0)

  X_df = pd.read_csv("dataset/" + baseFileName + ".csv", header=None, skiprows=1)
  y_df = X_df[0]
  X_df = X_df.drop([0], axis=1)
  # NUMBER_OF_TRAINING_SET_SIZE = 178778 # v1
  # NUMBER_OF_TRAINING_SET_SIZE = 178684 # v2
  # NUMBER_OF_TRAINING_SET_SIZE = 178641  # v3
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
  print(str(NUMBER_OF_TRAINING_SET_SIZE), str(x_train.shape), str(x_vtest.shape), str(x_validation.shape),
        file=fileResults)
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

  print(y_train.mean())

  column_maxes = X_df.max()
  vocab_size = max(column_maxes[len(column_maxes) - 1], column_maxes[len(column_maxes) - 2])
  input_length = x_train.shape[1]

  ''' 'hidden_activation' : 'selu',
                             'output_activation' : 'sigmoid',
                             'dropout_0' : 0.0,
                             'dropout_1': 0.0,
                             'dropout_2': 0.0,
                             'dropout_3': 0.0,
                             'layer_1': 400,
                             'layer_2': 400,
                             'layer_3': 400,
                             'opt_type' : "Adam",
                             'lr' : 0.001,
                             'beta_1' : 0.9,
                             'beta_2' : 0.999,
                             'epsilon' : 1e-03,
                             'amsgrad' : True 
  if isTensorflowSeluInputFile == True and tensorflowSeluInputFile != "":
    model2 = load_model(tensorflowSeluInputFile, custom_objects={ 'f1' : f1, 'recall' : recall, 'precision' : precision,})
  else :
    hidden_activation = 'selu'
    dropout_0 = 0.0
    layer_1 = 400
    dropout_1 = 0.0
    layer_2 = 400
    dropout_2 = 0.0
    layer_3 = 400
    dropout_3 = 0.0
    output_activation = 'sigmoid'
    model2 = Sequential()
    model2.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length))
    model2.add(Flatten())
    # model2.add(Dropout(dropout_0))
    model2.add(Dense(layer_1, activation=hidden_activation))
    # model2.add(Dropout(dropout_1))
    model2.add(Dense(layer_2, activation=hidden_activation))
    # model2.add(Dropout(dropout_2))
    model2.add(Dense(layer_3, activation=hidden_activation))
    # model2.add(Dropout(dropout_3))
    model2.add(Dense(1, activation='sigmoid'))
    opt_type = "Adam"
    lr = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-03
    amsgrad = True
    opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
    model2.compile(loss='binary_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy', f1, precision, recall])
    batch_size = 128
    epochs = 10
    history = model2.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(x_validation, y_validation))
    checkpoint_file_path = "selu-" + baseFileName + "-best_model.hdf5"
    if prefix != "":
      checkpoint_file_path = prefix + "-" + checkpoint_file_path
    model2.save(checkpoint_file_path)
  if isCreatePredictionForSimulator == True:
    Z_pred_keras2 = model2.predict(Z_test_csv)
    fileToPrintZ = open(predictionSamplesDFileBaseName + "selu" + "-" + prefix + 'prediction.out', "w",
                        encoding="utf-8")
    for prediction in Z_pred_keras2:
      fileToPrintZ.write('' + str(int(round(prediction[0]))) + '\n')
    fileToPrintZ.close()
  y_pred_keras2_validation = model2.predict(x_vtest).ravel()

  y_pred_keras2_test = []
  #fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras2)
  #auc_keras = auc(fpr_keras, tpr_keras)
  score = model2.evaluate(x_vtest, y_vtest, verbose=0)
  out_str = "" + str(score[1]) + " " + str(score[2]) + " " + str(score[3]) + " " + str(score[4])
  # out_str = "" + str(epochs) + " " + out_str
  #out_str = "" + str(es.stopped_epoch) + " " + out_str
  print('KerasValid2', score[1], score[2], score[3], score[4], file=fileResults)
  i = 1
  for x_test_i, y_test_i in zip(x_test, y_test):
    y_pred_keras2_test.append(model2.predict(x_test_i).ravel())
    score = model2.evaluate(x_test_i, y_test_i, verbose=0)
    # print(len(score), str(x_test_1.shape), str(y_test_1.shape), file=fileResults)
    out_str = out_str + " " + str(score[1]) + " " + str(score[2]) + " " + str(score[3]) + " " + str(score[4])
    print('KerasTest2' + str(i), score[1], score[2], score[3], score[4], file=fileResults)
    i += 1
  print(out_str, file=fileResults)
  '''
  '''                   'hidden_activation' : 'tanh',
                          'output_activation' : 'sigmoid',
                          'dropout_0' : 0.5,
                          'dropout_1': 0.5,
                          'dropout_2': 0.0,
                          'dropout_3': 0.0,
                          'layer_1': 350,
                          'layer_2': 100,
                          'layer_3': 100,
                          'opt_type' : "Adam",
                          'lr' : 0.001,
                          'beta_1' : 0.9,
                          'beta_2' : 0.999,
                          'epsilon' : 1e-03,
                          'amsgrad' : True'''
  if isTensorflow1InputFile == True and tensorflow1InputFile != "":
    model1 = load_model(tensorflow1InputFile, custom_objects={'f1': f1, 'recall': recall, 'precision': precision, })
  else:
    output_dim = 40
    hidden_activation = 'tanh'
    dropout_0 = 0.5
    layer_1 = 350
    dropout_1 = 0.5
    layer_2 = 100
    dropout_2 = 0.0
    layer_3 = 100
    dropout_3 = 0.0
    output_activation = 'sigmoid'
    model1 = Sequential()
    model1.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length))
    model1.add(Flatten())
    model1.add(Dropout(dropout_0))
    model1.add(Dense(layer_1, activation=hidden_activation))
    model1.add(Dropout(dropout_1))
    # model1.add(Dense(layer_2, activation=hidden_activation))
    # model1.add(Dropout(dropout_2))
    # model1.add(Dense(layer_3, activation=hidden_activation))
    # model.add(Dropout(dropout_3))
    model1.add(Dense(1, activation=output_activation))
    opt_type = "Adam"
    lr = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-03
    amsgrad = True
    opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
    model1.compile(loss='binary_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy', f1, precision, recall])
    print(str(vocab_size), str(output_dim), str(input_length))
    batch_size = 128
    epochs = 199
    earlyStopping = False
    checkpoint_file_path = "" + hidden_activation + baseFileName + "-best_model.hdf5"
    if prefix != "":
      checkpoint_file_path = prefix + "-" + checkpoint_file_path
    if earlyStopping == True:  # simple early stopping
      # simple early stopping
      es = EarlyStopping(monitor="val_loss", mode='min', verbose=1, patience=2)  # val_loss , patience=1
      mc = ModelCheckpoint(checkpoint_file_path, monitor="val_loss", mode='min', save_best_only=True, verbose=1)
      history = model1.fit(x_train, y_train,
                           batch_size=batch_size,
                           callbacks=[es, mc],
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_validation, y_validation))
      model1.load_weights(checkpoint_file_path)
    else:
      history = model1.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_validation, y_validation))
    model1.save(checkpoint_file_path)
  if isCreatePredictionForSimulator == True:
    fileToPrintZ = open(predictionSamplesDFileBaseName + "1" + "-" + prefix + 'prediction.out', "w",
                        encoding="utf-8")
    for i in range(0,NUMBER_OF_PRED_FILE) :
      gc.collect()
      fileNumberSuffix = "p" + str(i)
      Z_test_csv = pd.read_csv("partOfCsv_" + predSuffix +"/" + predictionSamplesDFileBaseName +"_" + fileNumberSuffix + ".csv", header=None)
      Z_pred_keras = model1.predict(Z_test_csv)
      for prediction in Z_pred_keras:
        fileToPrintZ.write('' + str(int(round(prediction[0]))) + '\n')
    fileToPrintZ.close()
  y_pred_keras_validation = model1.predict(x_vtest).ravel()

  y_pred_keras_test = []
  # fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
  # auc_keras = auc(fpr_keras, tpr_keras)
  print("Tensorflow1", layer_str(output_dim, layer_1, layer_2, layer_3),
        dropout_str(dropout_0, dropout_1, dropout_2, dropout_3),
        opt_type + "(lr=" + str(lr) + ",beta_1=" + str(beta_1) + ",beta_2=" + str(beta_2) + ",epsilon=" + str(
          epsilon) + ",amsgrad=" + str(amsgrad) + ")",
        hidden_activation, output_activation, file=fileResults)
  score = model1.evaluate(x_vtest, y_vtest, verbose=0)
  out_str = "" + str(score[1]) + " " + str(score[2]) + " " + str(score[3]) + " " + str(score[4])
  # out_str = "" + str(epochs) + " " + out_str
  # out_str = "" + str(es.stopped_epoch) + " " + out_str
  print('KerasValid1', score[1], score[2], score[3], score[4], file=fileResults)
  i = 1
  for x_test_i, y_test_i in zip(x_test, y_test):
    y_pred_keras_test.append(model1.predict(x_test_i).ravel())
    score = model1.evaluate(x_test_i, y_test_i, verbose=0)
    # print(len(score), str(x_test_1.shape), str(y_test_1.shape), file=fileResults)
    out_str = out_str + " " + str(score[1]) + " " + str(score[2]) + " " + str(score[3]) + " " + str(score[4])
    print('KerasTest1' + str(i), score[1], score[2], score[3], score[4], file=fileResults)
    i += 1
  print(out_str, file=fileResults)
  '''
   'hidden_activation' : 'relu',
                           'output_activation' : 'sigmoid',
                           'dropout_0' : 0.5,
                           'dropout_1': 0.5,
                           'dropout_2': 0.5,
                           'dropout_3': 0.5,
                           'layer_1': 1000,
                           'layer_2': 1000,
                           'layer_3': 1000,
                           'opt_type' : "Adam",
                           'lr' : 0.001,
                           'beta_1' : 0.9,
                           'beta_2' : 0.999,
                           'epsilon' : 1e-03,
                           'amsgrad' : True'''
  if isTensorflow2InputFile == True and tensorflow2InputFile != "":
    model2 = load_model(tensorflow2InputFile, custom_objects={'f1': f1, 'recall': recall, 'precision': precision, })
  else:
    output_dim = 100
    hidden_activation = 'relu'
    dropout_0 = 0.5
    layer_1 = 1000
    dropout_1 = 0.5
    layer_2 = 1000
    dropout_2 = 0.5
    layer_3 = 1000
    dropout_3 = 0.5
    output_activation = 'sigmoid'
    model2 = Sequential()
    model2.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length))
    model2.add(Flatten())
    model2.add(Dropout(dropout_0))
    model2.add(Dense(layer_1, activation=hidden_activation))
    model2.add(Dropout(dropout_1))
    # model2.add(Dense(layer_2, activation=hidden_activation))
    # model2.add(Dropout(dropout_2))
    # model2.add(Dense(layer_3, activation=hidden_activation))
    # model2.add(Dropout(dropout_3))
    model2.add(Dense(1, activation=output_activation))
    opt_type = "Adam"
    lr = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-03
    amsgrad = True
    opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
    model2.compile(loss='binary_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy', f1, precision, recall])
    print(str(vocab_size), str(output_dim), str(input_length))
    batch_size = 128
    epochs = 50
    earlyStopping = True
    checkpoint_file_path = ""+ hidden_activation + baseFileName + "-best_model.hdf5"
    if prefix != "":
      checkpoint_file_path = prefix + "-" + checkpoint_file_path
    if earlyStopping == True:  # simple early stopping
      # simple early stopping
      es = EarlyStopping(monitor="val_loss", mode='min', verbose=1, patience=2)  # val_loss , patience=1
      mc = ModelCheckpoint(checkpoint_file_path, monitor="val_loss", mode='min', save_best_only=True, verbose=1)
      history = model2.fit(x_train, y_train,
                           batch_size=batch_size,
                           callbacks=[es, mc],
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_validation, y_validation))
      model2.load_weights(checkpoint_file_path)
    else:
      history = model2.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_validation, y_validation))
    model2.save(checkpoint_file_path)
  if isCreatePredictionForSimulator == True:
    fileToPrintZ = open(predictionSamplesDFileBaseName + "2" + "-" + prefix + 'prediction.out', "w",
                        encoding="utf-8")
    for i in range(0,NUMBER_OF_PRED_FILE) :
      gc.collect()
      fileNumberSuffix = "p" + str(i)
      Z_test_csv = pd.read_csv("partOfCsv_" + predSuffix +"/" + predictionSamplesDFileBaseName +"_" + fileNumberSuffix + ".csv", header=None)
      Z_pred_keras2 = model2.predict(Z_test_csv)
      for prediction in Z_pred_keras2:
        fileToPrintZ.write('' + str(int(round(prediction[0]))) + '\n')
    fileToPrintZ.close()
  '''
  y_pred_keras2_validation = model2.predict(x_vtest).ravel()

  y_pred_keras2_test = []
  # fpr_keras2, tpr_keras2, thresholds_keras2 = roc_curve(y_test, y_pred_keras2)
  # auc_keras2 = auc(fpr_keras2, tpr_keras2)

  score = model2.evaluate(x_vtest, y_vtest, verbose=0)
  out_str = "" + str(score[1]) + " " + str(score[2]) + " " + str(score[3]) + " " + str(score[4])
  # out_str = "" + str(epochs) + " " + out_str
  # out_str = "" + str(es.stopped_epoch) + " " + out_str
  print('KerasValid2', score[1], score[2], score[3], score[4], file=fileResults)
  i = 1
  for x_test_i, y_test_i in zip(x_test, y_test):
    y_pred_keras2_test.append(model2.predict(x_test_i).ravel())
    score = model2.evaluate(x_test_i, y_test_i, verbose=0)
    # print(len(score), str(x_test_1.shape), str(y_test_1.shape), file=fileResults)
    out_str = out_str + " " + str(score[1]) + " " + str(score[2]) + " " + str(score[3]) + " " + str(score[4])
    print('KerasTest2' + str(i), score[1], score[2], score[3], score[4], file=fileResults)
    i += 1
  print(out_str, file=fileResults)
  

  if isCreatePredictionForSimulator == True:
    pred_array_Z = np.row_stack([Z_pred_keras, Z_pred_keras2, Z_pred_rf])
    pred_array_Z = pred_array_Z.mean(axis=0)
    pred_array_Z = pred_array_Z.round()
    fileToPrintZ = open(predictionSamplesDFileBaseName + prefix + 'prediction.out', "w", encoding="utf-8")
    for prediction in pred_array_Z:
      fileToPrintZ.write('' + str(int(prediction)) + '\n')
    fileToPrintZ.close()
  pred_array_valid = np.row_stack([y_pred_keras_validation, y_pred_keras2_validation, y_pred_rf_valid2])
  pred_array_valid = pred_array_valid.mean(axis=0)
  pred_array_valid = pred_array_valid.round()
  accuracy_voting_valid = accuracy_score(y_vtest, pred_array_valid)
  f1_voting_valid = f1_score(y_vtest, pred_array_valid)
  precision_voting_valid = precision_score(y_vtest, pred_array_valid)
  recall_voting_valid = recall_score(y_vtest, pred_array_valid)
  # fpr_voting_valid, tpr_voting_valid, thresholds_voting_valid = roc_curve(y_vtest, pred_array_valid)
  # auc_voting_valid = auc(fpr_voting_valid, tpr_voting_valid)
  outstr = "" + str(accuracy_voting_valid) + " " + str(f1_voting_valid) + " " + str(precision_voting_valid) + " " + str(
    recall_voting_valid)
  print('VotingValid', accuracy_voting_valid, f1_voting_valid, precision_voting_valid, recall_voting_valid,
        file=fileResults)
  i = 0
  for x_test_i, y_test_i in zip(x_test, y_test):
    # print(str(len(y_pred_keras_test[i])), str(len(y_pred_keras2_test[i])), str(len(y_pred_rf2[i])), file=fileResults)
    pred_array = np.row_stack([y_pred_keras_test[i], y_pred_keras2_test[i], y_pred_rf2[i]])
    pred_array = pred_array.mean(axis=0)
    pred_array = pred_array.round()
    accuracy_voting = accuracy_score(y_test_i, pred_array)
    f1_voting = f1_score(y_test_i, pred_array)
    precision_voting = precision_score(y_test_i, pred_array)
    recall_voting = recall_score(y_test_i, pred_array)
    # fpr_voting, tpr_voting, thresholds_voting = roc_curve(y_test_i, pred_array)
    # auc_voting = auc(fpr_voting, tpr_voting)
    outstr = outstr + " " + str(accuracy_voting) + " " + str(f1_voting) + " " + str(precision_voting) + " " + str(
      recall_voting)
    i = i + 1
    print('VotingTest' + str(i), accuracy_voting, f1_voting, precision_voting, recall_voting, file=fileResults)
  print(outstr, file=fileResults)

  fileResults.close()

  
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.25f})'.format(auc_keras))
  plt.plot(fpr_xgb, tpr_xgb, label='XGB (area = {:.25f})'.format(auc_xgb))
  plt.plot(fpr_dt, tpr_dt, label='DT (area = {:.25f})'.format(auc_dt))
  plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.25f})'.format(auc_rf))
  plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.25f})'.format(auc_lr))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig('roc1.png')
  # Zoom in view of the upper left corner.
  plt.figure(2)
  plt.xlim(0, 0.2)
  plt.ylim(0.8, 1)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.25f})'.format(auc_keras))
  plt.plot(fpr_xgb, tpr_xgb, label='XGB (area = {:.25f})'.format(auc_xgb))
  plt.plot(fpr_dt, tpr_dt, label='DT (area = {:.25f})'.format(auc_dt))
  plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.25f})'.format(auc_rf))
  plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.25f})'.format(auc_lr))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve (zoomed in at top left)')
  plt.legend(loc='best')
  plt.savefig('roc2.png')
  '''

