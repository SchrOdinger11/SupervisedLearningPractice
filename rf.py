import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 

def entropy(label):
    total = len(label)
    frequency_label = Counter(label)
    entropy = 0
    for frequency in frequency_label.values():
        probability = frequency / total
        if probability > 0:  
            entropy -= probability * np.log2(probability)
    return entropy

def InformationGain(feat, X, y):
    
    
    if X[feat].dtype.kind in 'if': 
        
        indices = np.argsort(X[feat])

        sortedTrain = X[feat].iloc[indices]
        sortedLabels = y.iloc[indices]

        maxGain = -1*np.infty
        maxSplit = None

        for i in range(1, len(sortedTrain)):


            if sortedTrain.iloc[i] == sortedTrain.iloc[i - 1]:
                continue  

            split = (sortedTrain.iloc[i] + sortedTrain.iloc[i-1]) / 2
            left = sortedLabels[sortedTrain <= split]
            right = sortedLabels[sortedTrain > split]

            leftSubset = entropy(left)
            rightSubset = entropy(right)
            featureEntropy = (len(left) / len(y)) * leftSubset + (len(right) / len(y)) * rightSubset

            infoGainn = entropy(y) - featureEntropy
            if infoGainn > maxGain:
                maxGain = infoGainn
                maxSplit = split


        return maxGain, maxSplit 
    else:  

        val = np.unique(X[feat])
        
        for value in val:
            subset = y[X[feat] == value]
            subset = entropy(subset)
            featureEntropy += (len(subset) / len(y)) * subset


        infoGainn = entropy(y) - featureEntropy
        return infoGainn, None  


class Node:
    def __init__(self, feature=None, featureIndex=None, value=None, split=None, isNumerical=False):
        self.feature = feature
        self.featureIndex = featureIndex
        self.value = value
        self.split = split  
        self.isNumerical = isNumerical  
        self.childNodes = {}

def decisionTree(X, y, depth=0, maxDepth=10):
    if len(np.unique(y)) == 1 or (depth >= maxDepth): 
        mostCommonValue = Counter(y).most_common(1)[0][0]
        return Node(value=mostCommonValue)

    totalFeatures = X.shape[1]
    m = int(np.sqrt(totalFeatures))  

    
    featuresToConsider = random.sample(X.columns.tolist(), m)

    bestFeature = None
    maxInfoGain = -1*np.inf
    bestSplit = None
    bestFeatureIndex = None
    isNumerical = False



    for featureName in featuresToConsider:
        infoGain, split = InformationGain(featureName, X, y)
        if infoGain > maxInfoGain:
            maxInfoGain = infoGain
            bestFeature = featureName
            bestSplit = split
            isNumerical = split is not None
            bestFeatureIndex = X.columns.get_loc(featureName)  

    if bestFeature is None:
        return Node(value=Counter(y).most_common(1)[0][0])

    node = Node(feature=bestFeature, featureIndex=bestFeatureIndex, split=bestSplit, isNumerical=isNumerical)

    if isNumerical:
        leftSplit = X[bestFeature] <= bestSplit
        rightSplit = X[bestFeature] > bestSplit
        node.childNodes['<='] = decisionTree(X[leftSplit], y[leftSplit], depth + 1, maxDepth)
        node.childNodes['>'] = decisionTree(X[rightSplit], y[rightSplit], depth + 1, maxDepth)
    else:
        for val in np.unique(X[bestFeature].values):
            subset = X[bestFeature] == val
            node.childNodes[val] = decisionTree(X[subset], y[subset], depth + 1, maxDepth)

    return node
def predict(node, sample, defaultLabel):

    if node.value is not None:

        return node.value

    if node.isNumerical == True:

        if sample[node.featureIndex] <= node.split:
            return predict(node.childNodes['<='], sample, defaultLabel)
        else:
            return predict(node.childNodes['>'], sample, defaultLabel)
    else:

        if sample[node.featureIndex] in node.childNodes:
            return predict(node.childNodes[sample[node.featureIndex]], sample, defaultLabel)
        else:

            return defaultLabel




def randomForest(X, y, ntree,max_depth=7):
    allDecisionTree = []
    for _ in range(ntree):
        sampledX, sampledY= bootstrapping(X, y)
        tree = decisionTree(sampledX, sampledY,0,max_depth)
        allDecisionTree.append(tree)

    return allDecisionTree

def majorityVotes(forest, sample,defaultLabel):
    predictions=[]
    for models in forest:

      predictions.append(predict(models,sample,defaultLabel))

    return Counter(predictions).most_common(1)[0][0]

def bootstrapping(X, y):
    indices = np.random.randint(0, len(X), size=len(X))
    return X.iloc[indices], y.iloc[indices]




def stratifiedCrossValidation(X, y, ntree, k=10):
  
    accuracies=[]
    precisions=[]
    recalls=[]
    f1s = []

   
    labels = np.unique(y)

    classIndex = {}


    for c in labels:
        indices = np.where(y == c)[0]
        classIndex[c] = indices


   
    foldsIneachClass = {}

    for label, index in classIndex.items():
        
        shuffle = np.random.permutation(index)
       
        folds = np.array_split(shuffle, k)
        
        foldsIneachClass[label] = folds



    
    for foldNumber in range(k):
        
        testIndices = []
        trainIndices = []

       
        for folds in foldsIneachClass.values():
            for index in folds[foldNumber]:
                testIndices.append(index)

        for folds in foldsIneachClass.values():
            for i in range(len(folds)): 
                if i != foldNumber:
                    for index in folds[i]:
                        trainIndices.append(index)
        X_train, y_train = X.iloc[trainIndices], y.iloc[trainIndices]
        X_test, y_test = X.iloc[testIndices], y.iloc[testIndices]
        fallBackLabel = Counter(y_train).most_common(1)[0][0]

        
        forest = randomForest(X_train, y_train, ntree)
        accuracy, precision, recall, f1 = calculateMetrics(X_test, y_test, fallBackLabel, forest)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s)



def calculateMetrics(xTest,yTest,fallBackLabel,forest):
  predictedLabel=[]
  actualLabels=[]
  for instance, actualLabel in zip(xTest.values,yTest.values):
    actualLabels.append(actualLabel)
    predictedLabel.append(majorityVotes(forest,instance,fallBackLabel))

  accuracy=calculateAccuracy(actualLabels,predictedLabel)
  precision=calculatePrecision(actualLabels,predictedLabel)
  recall=calculateRecall(actualLabels,predictedLabel)
  f1=2*precision*recall/(precision+recall)
  return accuracy,precision,recall,f1





def calculateAccuracy(actualLabels,predictedLabels):
    count=0
    for actual, prediction in zip(actualLabels,predictedLabels):
        if actual == prediction:
            count=count+1
    return count/len(predictedLabels)

def calculatePrecision(actualLabels, predictedLabels):
    labels = set(actualLabels)
    precisionSum = 0

    for c in labels:
        truePositives = 0
        totalPositives = 0
        for actual, prediction in zip(actualLabels, predictedLabels):
            if prediction == c:
                totalPositives += 1
                if actual == prediction:
                    truePositives += 1

       
        if totalPositives > 0:
            classPrecision = truePositives / totalPositives
            precisionSum += classPrecision


    
    return precisionSum / len(labels)


def calculateRecall(actualLabels, predictedLabels):
    labels = set(actualLabels)
    recallSum = 0

    for c in labels:
        truePositives = 0
        actualPositives = 0
        for actual, prediction in zip(actualLabels, predictedLabels):
            if actual == c:
                actualPositives += 1
                if prediction == c:
                    truePositives += 1

        
        if actualPositives > 0:
            classRecall = truePositives / actualPositives
            recallSum += classRecall

   
    
    return recallSum / len(labels)




#-------------Cancer and housevotes dataset --------------------------
#houseVotes=pd.read_csv('/content/hw3_cancer.csv',sep='\t',header=0)
#houseVotes=pd.read_csv('/content/hw3_house_votes_84.csv')

#X= houseVotes.iloc[:, :-1]
#Y= houseVotes.iloc[:, -1]

#--------------- Wine dataset------------------------------------------
#houseVotes=pd.read_csv('/content/hw3_wine.csv',sep='\t',header=0)

# X = houseVotes.iloc[:, 1:]
# Y = houseVotes.iloc[:, 0]





#For contraceptive dataset-------------------------------------------------
houseVotes = fetch_ucirepo(id=30) 
  
# data (as pandas dataframes) 
X = houseVotes.data.features 
Y = houseVotes.data.targets 
Y=Y.squeeze()
#--------------------------------------------------------



ntree=[1,5,10,20,30,40,50]
accuracy=[]
precision=[]
recall=[]
f1=[]

for n in ntree:

  a,p,r,f = stratifiedCrossValidation(X, Y, n, k=10)
  accuracy.append(a)
  precision.append(p)
  recall.append(r)
  f1.append(f)






figure, axis = plt.subplots(2, 2, figsize=(12, 10))  

print("Metric values for of n is ")
print(ntree)
print(accuracy)
print(precision)
print(recall)
print(f1)


axis[0, 0].plot(ntree, accuracy, marker='o', linestyle='-', color='blue')
axis[0, 0].set_title('Accuracy vs Number of Trees')
axis[0, 0].set_xlabel('Number of Trees')
axis[0, 0].set_ylabel('Accuracy')


axis[0, 1].plot(ntree, precision, marker='o', linestyle='-', color='red')
axis[0, 1].set_title('Precision vs Number of Trees')
axis[0, 1].set_xlabel('Number of Trees')
axis[0, 1].set_ylabel('Precision')


axis[1, 0].plot(ntree, recall, marker='o', linestyle='-', color='green')
axis[1, 0].set_title('Recall vs Number of Trees')
axis[1, 0].set_xlabel('Number of Trees')
axis[1, 0].set_ylabel('Recall')


axis[1, 1].plot(ntree, f1, marker='o', linestyle='-', color='magenta')
axis[1, 1].set_title('F1 Score vs Number of Trees')
axis[1, 1].set_xlabel('Number of Trees')
axis[1, 1].set_ylabel('F1 Score')


plt.tight_layout()

plt.show()




