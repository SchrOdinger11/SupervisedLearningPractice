import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

houseVotes=pd.read_csv('house_votes_84.csv')

def entropy(label):
  FrequencyLabel = Counter(label)
  total=len(label)

  entropy=0
  for label, frequency in FrequencyLabel.items():

    entropy=entropy+ -1*(frequency/total)*np.log2(frequency/total)

  return entropy
def InformationGain(feat,train,y):
  maxEntropy=entropy(y)

  labelFeature = np.unique(train[feat])
  featureEntropy=0
  print(labelFeature)
  for labelFeatureInstance in labelFeature:
    
    frequency=train[feat]==labelFeatureInstance
    totalOfInstance=np.sum(frequency)

    SubsetEntropy=entropy(y[frequency])

    featureEntropy=featureEntropy+(totalOfInstance/len(y))*SubsetEntropy



  return maxEntropy - featureEntropy

def giniIndex(label):
    
    if len(label) == 0:
        return 0

    FrequencyLabel = Counter(label)
    gini=1
    total=len(label)
    for label, frequency in FrequencyLabel.items():

      gini =gini- frequency / total ** 2

    return gini
def GiniIndexGain(feat,train,y):
    maxGini = giniIndex(y)

    labelFeature = np.unique(train[feat])
    featureGini = 0

    for labelFeatureInstance in labelFeature:
        frequency = train[feat] == labelFeatureInstance
        totalOfInstance = np.sum(frequency)

        SubsetGini = giniIndex(y[frequency])

        featureGini = featureGini + (totalOfInstance / len(y)) * SubsetGini

    return maxGini-featureGini


#print(InformationGain("handicapped-infants",xTrain,yTrain))


class Node:
    def __init__(self, feature=None,featureIndex=None, value=None):
        self.feature = feature
        self.featureIndex=featureIndex
        self.value = value
        self.childNodes = {}

def eightFivepercentCriteriaSatisfied(y):
  
  if max(Counter(y).values()) / len(y) >= 0.85:
    print("yo ",max(Counter(y).values()))
    return True
  else:
    return False
def decisionTree(X,y):
  #corner condition when label has only one type of unique value

    if len(np.unique(y)) == 1 :
        mostCommonValue=Counter(y).most_common(1)[0][0]
        return Node(value=mostCommonValue)

#Uncomment this for running the additional criteria
    # if eightFivepercentCriteriaSatisfied(y):
    #     mostCommonValue=Counter(y).most_common(1)[0][0]
    #     return Node(value=mostCommonValue)

    totalFeatures = X.shape[1]
    nextFeature=None
    maxInfoGain =  -1*np.infty
    minGiniGain= -1*np.infty
    featureNames = X.columns.tolist()
    nextFeatureIndex=0


    
  # #build DT here

    for eachFeatureIndex in range(totalFeatures):

#comment this for and uncomment below lines for running Gini
      infoGain=InformationGain(featureNames[eachFeatureIndex],X,y)
      if(infoGain>maxInfoGain):
        maxInfoGain=infoGain
        nextFeature=featureNames[eachFeatureIndex]
        nextFeatureIndex=eachFeatureIndex
      

      #Uncomment this for running Gini
      # giniGain=GiniIndexGain(featureNames[eachFeatureIndex],X,y)
      # if(giniGain>minGiniGain):
      #   minGiniGain=giniGain
      #   nextFeature=featureNames[eachFeatureIndex]
      #   nextFeatureIndex=eachFeatureIndex




    node= Node(feature= nextFeature, featureIndex= nextFeatureIndex)


    uniqueValues = np.unique(X[nextFeature].values)

  # #now for the next feature i.e. feature with highest info gain you will check the values it can take
    for val in uniqueValues:
      subset = X[nextFeature] == val
      node.childNodes[val] = decisionTree(X[subset], y[subset])

    return node

def predict(node, instance):
    if node.value is not None:
        return node.value

    if instance[node.featureIndex] in node.childNodes:
        return predict(node.childNodes[instance[node.featureIndex]], instance)

    mostCommonValue = Counter(instance).most_common(1)[0][0]
    return mostCommonValue


accuracyTraining=[]
accuracyTesting=[]

for i in range (100):
  houseVotes=sk.utils.shuffle(houseVotes)

  X= houseVotes.iloc[:, :-1]
  #print(np.unique(train["handicapped-infants"]))
  Y= houseVotes.iloc[:, -1]
  NormalizedFeature = X
  xTrain, xTest, yTrain, yTest = train_test_split(NormalizedFeature, Y,  test_size = 0.2)

  rootNode = decisionTree(xTrain,yTrain)


  yPrediction=[]

  for i, row in xTrain.iterrows():  #to check for test replace xTrain with xTest

    yPrediction.append(predict(rootNode,row))


  #predictions_test = [predict(rootNode, sample) for sample in xTest]
  
  accuracyTrain = np.mean(yPrediction == yTrain)   #to check for test replace yTrain with yTest
  accuracyTesting.append(accuracyTrain)


#plt.subplot(1, 2, 2)
plt.hist(accuracyTesting, bins=10, color='salmon', edgecolor='black')
print("Average accuracy is ",np.mean(accuracyTesting))
print("Standard Deviation is ",np.std(accuracyTesting))
plt.title('Accuracy Distribution on Training Set')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

