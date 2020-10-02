import numpy as np
import sys

isCreatePredictionForSimulator = False
predictionSamplesDFileBaseName = "1010rr1101rr10network100000"
prefix = ""
isRandomForestInputFile = False
randomForestInputFile = ""
isTensorflow1InputFile = False
tensorflow1InputFile = ""
isTensorflow2InputFile = False
tensorflow2InputFile = ""
if len(sys.argv) > 1:
  prefix = str(sys.argv[1])
if len(sys.argv) > 2:
  isCreatePredictionForSimulator = True
  predictionSamplesDFileBaseName = str(sys.argv[2])
if len(sys.argv) > 3:
  isRandomForestInputFile = True
  randomForestInputFile = str(sys.argv[3])
if len(sys.argv) > 4:
  isTensorflow1InputFile = True
  tensorflow1InputFile = str(sys.argv[4])
if len(sys.argv) > 5:
  isTensorflow2InputFile = True
  tensorflow2InputFile = str(sys.argv[5])


predictionsRF = []
with open(randomForestInputFile) as f1:
  for l1 in f1:
    predictionsRF.append(int(str(l1).split('\n')[0]))

predictionsTF1 = []
with open(tensorflow1InputFile) as f2:
  for l2 in f2:
    predictionsTF1.append(int(float(str(l2).split('\n')[0])))

predictionsTF2 = []
with open(tensorflow2InputFile) as f3:
  for l3 in f3:
    predictionsTF2.append(int(float(str(l3).split('\n')[0])))

pred_array_Z = np.row_stack([predictionsTF1,predictionsTF2,predictionsRF])
pred_array_Z = pred_array_Z.mean(axis=0)
pred_array_Z = pred_array_Z.round()
fileToPrintZ = open(predictionSamplesDFileBaseName+prefix+'prediction.out', "w", encoding="utf-8")
for prediction in pred_array_Z :
  fileToPrintZ.write(''+str(int(prediction))+'\n')
fileToPrintZ.close()