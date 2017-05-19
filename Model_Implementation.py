from SVM import CrossValidation, InitializingSVC
from sklearn import svm, metrics, feature_selection, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
import scipy as sp
from math import inf
import random
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    Features = []
    Labels = []

    with open("Features.csv", 'r') as Features_File:
        for line in Features_File:
            Features.append(line.split(','))
    print(len(Features))
    

    with open("Labels.csv", 'r') as Labels_File:
        for line in Labels_File:
            Labels = line.split(',')
    print(len(Labels))
    randomGen = random.random()
    random.shuffle(Features, lambda : randomGen)
    random.shuffle(Labels, lambda : randomGen)
    Labels = sp.array(Labels)
    Features = sp.array(Features)

    print("=============RANDOM FOREST=================") 
    clf = RandomForestClassifier(n_estimators=10)
    scores_Forest = cross_val_score(clf, Features, Labels, cv = 10)
    print(scores_Forest)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores_Forest.mean(), scores_Forest.std() * 2))
    
    Labels = sp.array(Labels)
    print(Labels.shape)
    print(Features.shape)
    print("=============SVM=================") 
   # clf = ExtraTreesClassifier()
   # clf.fit(Features, Labels)
   # print(sp.ndarray.tolist(clf.feature_importances_))
    Score = CrossValidation(Features, Labels, Gamma=1/250, C = 3)
    print(Score, Score.mean())
