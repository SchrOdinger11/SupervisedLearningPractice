import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

iris=pd.read_csv('iris.csv') 

def euclidieanDistance(i,t):

  distance= np.linalg.norm(np.array(i) - np.array(t))

  return distance

def knnPredict(test,Ylabel,k):

  labelForTest=[]
  checkForTypeOfDataset=False
  if len(test)==30:
    checkForTypeOfDataset= True
  for i in range(len(test)):
    #print(i)
    #print("first test is" ,test[i])
    distanceFromTrainData=[]
    for t in range(len(xTrainNormalized)):
      # if(checkForTypeOfDataset==False ):
      #   if (i!=t):
      # #print("first train is" ,xTrain[t])
      #     ed=euclidieanDistance(test[i],xTrain[t]) #xTrain
      #     distanceFromTrainData.append([ed,t])
      #if (checkForTypeOfDataset==True):
      ed=euclidieanDistance(test[i],xTrainNormalized[t]) #xTrain
      distanceFromTrainData.append([ed,t])
    distanceFromTrainData.sort()

    #print(type(distanceFromTrainData))

    kDistances=distanceFromTrainData[:k]
    labelVector=[]
    for instances in range(len(kDistances)):
      #print("Helli")
      label= yTrain[kDistances[instances][1]]
      #print (kDistances[instances])
      labelVector.append(label)

    labelCount=Counter(labelVector)
    PredictedLabel=labelCount.most_common(1)[0][0]
    labelForTest.append(PredictedLabel)
  #print(len(labelForTest))

  accuracy = np.mean(labelForTest == Ylabel)
  #print(accuracy)
  # for i in range(len(labelForTest)):
  #   print("Predicted ", labelForTest[i]," Actual is ", yTrain[i])

  return accuracy



accuracyTraining=[]
accuracyTesting=[]
standardDeviationTrain=[]
standardDeviationTest=[]
for k in range (1,52,2):

  idx=0

  accuracyTrain=[]
  accuracyTest=[]

  for x in range (20):

    iris= sk.utils.shuffle(iris)
    #print(len(iris))
    #train, test = train_test_split(iris, test_size=0.2)
    #print(train)
    #print("test")
    #print(len(test))
    irisData=np.array(iris)
    X= irisData[:,:-1]
    Y= irisData[:,-1]

    xTrain, xTest, yTrain, yTest = train_test_split(X, Y,  test_size = 0.2)
    xTrainNormalized = (xTrain - np.min(xTrain, axis=0)) / (np.max(xTrain, axis=0) - np.min(xTrain, axis=0))


    xTestNormalized= (xTest-np.min(xTrain,axis=0))/(np.max(xTrain,axis=0)-np.min(xTrain,axis=0))

    iterationAccuracyTrain=knnPredict(xTrainNormalized,yTrain,k)
    iterationAccuracyTest=knnPredict(xTestNormalized,yTest,k)

    #print(k,iterationAccuracyTrain)
    accuracyTrain.append(iterationAccuracyTrain)
    accuracyTest.append(iterationAccuracyTest)
  standardDeviationTrain.append(np.std(accuracyTrain))
  standardDeviationTest.append(np.std(accuracyTest))
  accuracyTraining.append(np.mean(accuracyTrain))
  accuracyTesting.append(np.mean(accuracyTest))




# xTrain=train.iloc[:,0: 3]
# yTrain=train.iloc[:, -1]

# xTest=test.iloc[:,0: 3]
# yTest=test.iloc[:, -1]




#0.9663865546218487 - train
#0.9333333333333333 - test
k = list(range(1, 52, 2))
plt.errorbar(k, accuracyTraining, yerr=standardDeviationTrain, label='Training Accuracy',ecolor='red')


plt.xlabel('k Values')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(k)
plt.scatter(k, accuracyTraining, color='blue', marker='o')
plt.title('Accuracy vs k Values for k-NN Model')
plt.show()
plt.errorbar(k, accuracyTesting, yerr=standardDeviationTest, label='Testing Accuracy',ecolor='red')

plt.xlabel('k Values')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(k)
plt.scatter(k, accuracyTesting, color='blue', marker='o')
plt.title('Accuracy vs k Values for k-NN Model')
plt.show()