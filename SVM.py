from Model import CrossValidation, InitializingSVC
from sklearn import svm, metrics, feature_selection, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
import scipy as sp
from math import inf

if __name__ == "__main__":
    Features = []
    Labels = []

    with open("Features.csv", 'r') as Features_File:
        for line in Features_File:
            Features.append(line.split(','))
    print(len(Features[0]))
    Features = sp.array(Features)

    with open("Labels.csv", 'r') as Labels_File:
        for line in Labels_File:
            Labels = line.split(',')

    Labels = sp.array(Labels)
    print(Labels.shape)
    print(Features.shape)
    clf = ExtraTreesClassifier()
    clf.fit(Features, Labels)
    print(sp.ndarray.tolist(clf.feature_importances_))
    Score = CrossValidation(Features, Labels, Gamma=1/250, C = 3)
    print(Score, Score.mean())