from utils import load_training_set,load_test_set
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import random


def multinomialNaiveBayes(pReviews, nReviews,dataStructure,a):
    positiveFreqList={}
    negativeFreqList={}

    for list in pReviews:
        for words in list:
            positiveFreqList[words]=positiveFreqList.get(words,0)+1
    
    for list in nReviews:
        for words in list:
            negativeFreqList[words]=negativeFreqList.get(words,0)+1

    p={}

    pClass={}
    pClass['pos']=len(pReviews)/(len(pReviews)+len(nReviews))
    pClass['neg']=len(nReviews)/(len(pReviews)+len(nReviews))
    if '-1' not in p:
        p['-1'] = {}
    #calculate P(wk|yi)
    wordling=''



    for word in dataStructure:
        wordling = word
        # print("dekh")
        # print(a+positiveFreqList.get(word, 0))
        # print(denominator)
        numerator=positiveFreqList.get(word, 0)  +a
        denominator= sum(positiveFreqList.values()) + a* len(dataStructure)

   
        numerNeg=  negativeFreqList.get(word, 0)+a
        denomNeg=sum(negativeFreqList.values()) + a* len(dataStructure)
        p[word] = {"pos": numerator / denominator,
                "neg":numerNeg /denomNeg}

        p['-1']['pos']= a/denominator
        p['-1']['neg']= a/denomNeg

   
    return p,pClass
    #calculate P(wk|yi='postive')
        


def predictClass(p, pClass, reviews, dataStructure):
    probability = {}
# #Q1A
    # for yi in pClass:
    #     probability[yi] = pClass[yi]
    #     for word in reviews:
    #         if word in p :
    #              probability[yi]  *= p[word][yi]
    #         else:
    #             probability[yi]  *= 1e-8
#Q1B
    for yi in pClass:
        if yi!='-1':
            probability[yi] = np.log(pClass[yi])

            for word in reviews:
                if word in p :
                    probability[yi]  += np.log(p[word][yi])
                else:
                    probability[yi] += p['-1'][yi]
            # else:
            #     probability[yi]  += np.log(1e-8)


    
    #print(probability)
    posKey=probability['pos']
    negKey=probability['neg']
    if posKey==negKey:
        key= random.choice(['pos', 'neg'])
    else:
        if(posKey>negKey):
            key='pos'
        else:
            key='neg'



    return key

def testInstances(pTest,nTest,p,pClass,dataStructure):
    predictedLabels=[]
    actualLabels=[]

    for reviews in pTest:
        actualLabels.append("pos")
        
        x=predictClass(p,pClass,reviews,dataStructure)
        predictedLabels.append(x)
       
    
    for reviews in nTest:
        actualLabels.append("neg")
        x=predictClass(p,pClass,reviews,dataStructure)
        predictedLabels.append(x)
       


    accuracy=calculateAccuracy(actualLabels,predictedLabels)
    precision=calculatePrecision(actualLabels,predictedLabels)
    recall = calculateRecall(actualLabels,predictedLabels)
    print("Confusion Matrix:")
    confusionMatrix = calculateConfusionMatrix(actualLabels,predictedLabels)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
   
    

    return accuracy,precision,recall,confusionMatrix

def calculateAccuracy(actualLabels,predictedLabels):
    count=0
    for actual, prediction in zip(actualLabels,predictedLabels):
        if actual == prediction:
            count=count+1
    return count/len(predictedLabels) 

def calculatePrecision(actualLabels,predictedLabels):
    positiveLabel='pos'
    truePositives=0
    totalPositives=0
    for actual, prediction in zip(actualLabels,predictedLabels):
        if actual==prediction==positiveLabel:
            truePositives=truePositives+1

        if prediction==positiveLabel:
            totalPositives=totalPositives+1
    return truePositives/totalPositives

def calculateRecall(actualLabels,predictedLabels):
    positiveLabel='pos'
    truePositives=0
    actualPositives=0
    for actual, prediction in zip(actualLabels,predictedLabels):
        if actual==prediction==positiveLabel:
            truePositives=truePositives+1

        if actual==positiveLabel:
            actualPositives=actualPositives+1
    return truePositives/actualPositives

def calculateConfusionMatrix(actualLabels,predictedLabels):
    truePositive=0
    falsePositive=0
    trueNegative=0
    falseNegative=0
    for actual, prediction in zip(actualLabels,predictedLabels):
        if(actual==prediction=='pos'):
            truePositive=truePositive+1
        if(prediction=='pos' and actual =='neg'):
            falsePositive=falsePositive+1
        if(actual == prediction == 'neg'):
            trueNegative=trueNegative+1
        if(actual == 'pos' and prediction =='neg'):
            falseNegative=falseNegative+1
    print(f'            Predicted Pos  Predicted Neg')
    print(f'Actual Pos       {truePositive}                  {falseNegative}')
    print(f'Actual Neg       {falsePositive}                  {trueNegative}')

def main():
    positiveProcessedReviews, negativeProcessedReviews,dataStructure=load_training_set(0.5,0.5)
    print(len(positiveProcessedReviews+negativeProcessedReviews))
    a=10
    probability, probabilityClass=multinomialNaiveBayes(positiveProcessedReviews,negativeProcessedReviews,dataStructure,a)

    TestpositiveProcessedReviews, TestnegativeProcessedReviews=load_test_set(1,1)

    testInstances(TestpositiveProcessedReviews,TestnegativeProcessedReviews,probability,probabilityClass,dataStructure)
    
    #plot(positiveProcessedReviews, negativeProcessedReviews,dataStructure)

def plot(positiveProcessedReviews,negativeProcessedReviews,dataStructure):
    a = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    accuracy=[]
    for aValues in a:
   
        TestpositiveProcessedReviews, TestnegativeProcessedReviews=load_test_set(0.2,0.2)
        probability, probabilityClass=multinomialNaiveBayes(positiveProcessedReviews,negativeProcessedReviews,dataStructure,aValues)
        a1,b1,c1,d1=testInstances(TestpositiveProcessedReviews,TestnegativeProcessedReviews,probability,probabilityClass,dataStructure)
        accuracy.append(a1)

    
    plt.plot(a, accuracy, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.xlabel('α (Log Scale)')
    plt.ylabel('Accuracy on Test Set')
    plt.title('Effect of α on Model Accuracy')
    plt.show()

if __name__ == "__main__":
    main()
#print(predictClass(probability,probabilityClass,text,dataStructure))