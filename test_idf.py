from FeatureEngineering import tf_idf_train, tf_idf_predict, ProcessCSV
import random

if __name__ == "__main__":
    Train_URLs = ProcessCSV("URL.txt")
    Vectorizer, FileName = tf_idf_train(Train_URLs)
    
    URLs = [URL[0] for URL in Train_URLs]

    URL_Predicts = tf_idf_predict(Vectorizer, URLs, FileName)
    URL_Original = [URL[1] for URL in Train_URLs[100:150]]
    Count = 0
    for i in range(len(URL_Original)):
    		if (URL_Original[i] != URL_Predicts[i]):
    				Count+= 1

    print(Count/50) 