# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:59:09 2017

@author: shikhar
"""

from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score
import scipy as sp
from math import inf

def NormalizingFeatures(File):
    """
    Transforming the file into feature vectors. Normalizing them into the range [-1,1].
    """
    FileNormalizing = open(File)
    Label = []
    Features = []
    for line in FileNormalizing:
        line = line.strip().split(',')
        Label.append(int(line[0]))
        Features.append([((2*int(i) / 255)-1) for i in line[1:]])
    Features = sp.array(Features)
    Label = sp.array(Label)
    FileNormalizing.close()
    return Label, Features

def InitializingSVC(User_Gamma = inf, User_C = inf):
    """
    Initializes SVC depending on the given value of Gamma and C.
    """
    if User_C != inf and User_Gamma != inf:
        Classifier = svm.SVC(C = User_C, gamma = User_Gamma)
    elif User_C == inf and User_Gamma != inf:
        Classifier = svm.SVC(gamma = User_Gamma)
    elif User_C != inf and User_Gamma == inf:
        Classifier = svm.SVC(C = User_C)
    else:
        Classifier = svm.SVC()
    return Classifier

def Predicting(TrainingFeatures, TrainingLabel, Test_Features, Test_Label, Gamma = inf, C = inf):
    """
    Uses given settings to predict the test error rate.
    """
    Errors = 0
    Classifier = InitializingSVC(Gamma, C)
    Classifier.fit(Training_Features, Training_Label)
    Expected = Test_Label
    Predicted = Classifier.predict(Test_Features)
    print("Classification report for classifier %s:\n%s\n"
      % (Classifier, metrics.classification_report(Expected, Predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(Expected, Predicted))
    for i in range(len(Predicted)):
        if Predicted[i] != Expected[i]:
            Errors += 1
    return (Errors/len(Predicted))*100

def CrossValidation(TrainingFeatures, TrainingLabel, Test_Features=[], Test_Label=[], Gamma = inf, C = inf):
    """
    Uses given settings to predict the cross validation score.
    """
    if (Test_Features != [] and Test_Label != []):
        Label = sp.ndarray.tolist(Training_Label) + sp.ndarray.tolist(Test_Label)
        Features = sp.ndarray.tolist(Training_Features) + sp.ndarray.tolist(Test_Features)
        Label = sp.array(Label)
        Features = sp.array(Features)
    else:
        Features = sp.array(TrainingFeatures)
        Label = sp.array(TrainingLabel)
    Classifier = InitializingSVC(Gamma, C)
    scores = cross_val_score(Classifier, Features, Label, cv=10)
    return scores


if __name__ == '__main__':
    TrainingFile = "mnist_train.txt"
    TestFile = "mnist_test.txt"
    Training_Label, Training_Features = NormalizingFeatures(TrainingFile)
    Test_Label, Test_Features = NormalizingFeatures(TestFile)

    print("\n\n\nDefault Settings of Gamma and C: \n\n")
    ErrorRate = Predicting(Training_Features, Training_Label, Test_Features, Test_Label)
    CrossValidation_ErrorRate = CrossValidation(Training_Features, Training_Label, Test_Features, Test_Label)
    print("\nTest Error Rate : {} ".format(ErrorRate))
    print("Cross Validation Score : {} ".format(CrossValidation_ErrorRate))
    print("Cross Validation Error Rate : {} ".format((1 - CrossValidation_ErrorRate.mean()) * 100))

    print("\n\n\nBetter Settings of Gamma and C, with Gamma = 1/150 and C = 3: \n\n")
    ErrorRate = Predicting(Training_Features, Training_Label, Test_Features, Test_Label, C = 3, Gamma = 1/150)
    CrossValidation_ErrorRate = CrossValidation(Training_Features, Training_Label, Test_Features, Test_Label, C = 3, Gamma = 1/150)

    print("\nTest Error Rate : {} ".format(ErrorRate))
    print("Cross Validation Score : {} ".format(CrossValidation_ErrorRate))
    print("Cross Validation Error Rate : {} ".format((1 - CrossValidation_ErrorRate.mean()) * 100))
