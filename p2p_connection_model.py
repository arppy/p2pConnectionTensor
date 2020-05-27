from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

import time
import os

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

baseFileName = "111111111111"
core = "3"
prefix = ""
if len(sys.argv) > 1 :
  baseFileName = str(sys.argv[1])
if len(sys.argv) > 2 :
  core = str(int(sys.argv[2])-1)
if len(sys.argv) > 3 :
  prefix = str(sys.argv[3])
os.environ["CUDA_VISIBLE_DEVICES"] = core
fileResults = open("nemb"+prefix+""+baseFileName + ".results", "w")

tf.reset_default_graph()
tf.set_random_seed(0)
np.random.seed(0)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333, allocator_type="BFC", visible_device_list="0")
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, allow_soft_placement=True, log_device_placement=True, inter_op_parallelism_threads=1, gpu_options=gpu_options, device_count={'GPU': 1})
#config = tf.ConfigProto()
##config.gpu_options.allow_growth = True
#config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
with sess.as_default():

    tf.set_random_seed(0)
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
        # print(str(x_test[i].shape), str(y_test[i].shape))
        i += 1

    model = Sequential()
    input_dim = x_train.shape[1]
    batch_size = 128
    epochs = 10
    model.add(Dense(400, activation='selu', input_dim=input_dim))
    #model.add(Dropout(0.5))
    model.add(Dense(400, activation='selu'))
    #model.add(Dropout(0.5))
    model.add(Dense(400, activation='selu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-03, amsgrad=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy',f1,precision,recall])
    #print('++++++++++++++++Model compile is over!')
    # simple early stopping
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        #          callbacks=[es],
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_validation, y_validation))
    #y_pred_keras = model.predict(x_validation).ravel()
    #fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_validation, y_pred_keras)
    #auc_keras = auc(fpr_keras, tpr_keras)
    score = model.evaluate(x_train, y_train, verbose=0)
    print('KerasTrain',score[1],score[2],score[3],score[4],file=fileResults)
    score = model.evaluate(x_validation, y_validation, verbose=0)
    print('KerasValid',score[1],score[2],score[3],score[4],file=fileResults)
    outstr = "" + str(score[1]) + " " + str(score[2]) + " " + str(score[3]) + " " + str(score[4])
    i = 1
    for x_test_i, y_test_i in zip(x_test, y_test):
        score = model.evaluate(x_test_i, y_test_i, verbose=0)
        # print(len(score), str(x_test_1.shape), str(y_test_1.shape), file=fileResults)
        outstr = outstr + " " + str(score[1]) + " " + str(score[2]) + " " + str(score[3]) + " " + str(score[4])
        print('KerasTest' + str(i), score[1], score[2], score[3], score[4], file=fileResults)
        i += 1
    print(outstr, file=fileResults)
    fileResults.close()

    '''
    y_pred_keras = model.predict(x_test).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)


    model_xgb = XGBClassifier()
    model_xgb.fit(x_train, y_train.values.ravel())
    # make predictions for test data
    y_pred_xgb = model_xgb.predict(x_test)
    predictions_xgb = y_pred_xgb
    #predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy_xgb= accuracy_score(y_test, predictions_xgb)
    f1_xgb= f1_score(y_test, predictions_xgb)
    precision_xgb = precision_score(y_test, predictions_xgb)
    print('Feature Set            | Accuracy                         | F1 measure                       |  Precesion')
    print('XGBoost                |', accuracy_xgb, '|', f1_xgb, '|', precision_xgb, '|', )
    fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_pred_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)


    rf = RandomForestClassifier(max_depth=10, n_estimators=100)
    rf.fit(x_train, y_train.values.ravel())
    y_pred_rf = rf.predict(x_test)
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
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.25f})'.format(auc_keras))
    plt.plot(fpr_xgb, tpr_xgb, label='XGB (area = {:.25f})'.format(auc_xgb))
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
    plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.25f})'.format(auc_rf))
    plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.25f})'.format(auc_lr))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.savefig('roc2.png')
    '''