from urllib.parse import urlparse
import re
import pandas as pd
import numpy as np
from sklearn import feature_extraction, model_selection, linear_model
import random
import pickle
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pygeoip
NOT_FOUND = -1


def ProcessCSV(FileName):
    """
    Reads a CSV File, and converts into list, with label being integer
    """
    Train_URLs = []
    for line in open("URL.txt"):
        line = line.split(',')
        line[1] = int(line[1])
        Train_URLs.append(line)
    return Train_URLs

def MakeTokens(URL):
    Tokens = re.split('\W+', URL)
    if 'com' in Tokens:
        Tokens.remove('com')
    return Tokens

def LexicographicalFeatures(URL):
    if URL == '':
        return ValueError

    NumberOfLevels = URL.count('.')

    URL = MakeTokens(URL) #Makes tokens
    URL = list(filter(None, URL)) #Removes empty strings
    LongestToken = max(URL, key = len)
    TotalLength = sum(len(Token) for Token in URL)
    NumberOfTokens = len(URL)

    return [TotalLength/NumberOfTokens, LongestToken, NumberOfLevels, NumberOfTokens]


def tf_idf_train(URLs):
    """
    Make On - core learning
    """
    FileName = "Logistic_Reg_Feature.sav" #To save the trained model

    random.shuffle(URLs)
    Y = [d[1] for d in URLs] #Stores labels, -1, or 1
    Corpus = [d[0] for d in URLs] #URLs
    Vectorizer = feature_extraction.text.TfidfVectorizer(tokenizer = MakeTokens)
                                    #Uses self made tokenizer function
    X = Vectorizer.fit_transform(Corpus)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
            X, Y, test_size = 0.1, random_state=42) #80-20 split
    LogisticRegression = linear_model.LogisticRegression()
    LogisticRegression.fit(X_train, Y_train)
    pickle.dump(LogisticRegression, open(FileName, 'wb'))
    return Vectorizer, FileName


def tf_idf_predict(Vectorizer, URL, FileName):
    """
    Uses Vectorizer function instantiated in the training, URL to predict and
    filename from which the Trained Model needs to be loaded
    """
    LogisticRegression = pickle.load(open(FileName, 'rb'))
    URL = Vectorizer.transform(URL)
    Y_Predict = LogisticRegression.predict(URL)
    return Y_Predict #Returns an array of the predicted label of URL


def AlexaRanking(URL):
    """
    Fix countries of non-malware
    """
    Website_Country = []
    soup = BeautifulSoup(urlopen("http://data.alexa.com/data?cli=10&dat=snbamz&url="+URL[0]).read(), "lxml")
    print(URL[1] , ":", URL[0], end = ' ')
    try:
        Website_Country.append(soup.popularity['text'])
    except:
        Website_Country.append(NOT_FOUND)
    try:
        Website_Country.append(soup.country['rank'])
    except:
        Website_Country.append(NOT_FOUND)
    print(Website_Country)
    return Website_Country


def SecuritySensitive(URL_Tokens):
    SecurityWords = ['confirm', 'account', 'banking', 'secure', 'ebayisapi', 'webscr', 'login', 'signin', 'verification']
    Count = 0
    for Token in URL_Tokens:
        for Words in SecurityWords:
            if Words in Token:
                Count += 1
    print(Count)
    return Count


def ExecutableURL(URL_Tokens):
    print(URL_Tokens)
    return int('exe' in URL_Tokens)


def IP_URL(URL_Tokens):
    Count = 0
    for Token in URL_Tokens:
        if Token.isnumeric():
            Count += 1
        else:
            Count = 0

        if Count >= 4:
            return 1
    return 0

def getASNumber(Host):
    try:
        File = pygeoip.GeoIP('GeoIPASNum.dat')
        ASN = int(File.org_by_name(Host).split()[0][2:])
        return ASN
    except:
        return NOT_FOUND
def Count_Function():


def MakeFeatures(URL, tf_idf_Vectorizer, tf_idf_FileName):
    URL_Tokens = MakeTokens(URL[:]) #Tokens of the URL
    print(URL)
    URL_Details = urlparse(URL[0]) # Makes the object of a URL to access different info
    Host = URL_Details.netloc
    Path = URL_Details.path

    Features = []
    Features.extend(LexicographicalFeatures(URL))
    #Avg length of tokens, Longest Token, Number of Levels, Number of Tokens
    Features.extend([len(MakeTokens(Path)), len(Path), len(Host)])
    #Number of tokens in path, length of path, length of Host
    Features.extend(AlexaRanking(URL))
    #Rank of URL, Rank of Country hosting URL
    Features.extend(SecuritySensitive(URL_Tokens))
    #Integer value if the tokens contain words indicative of malware
    Features.extend(ExecutableURL(URL_Tokens))
    #Binary if '.exe' exists in tokens or not
    Features.extend(IP_URL(URL_Tokens))
    #Binary if url contains IP Address
    Features.extend(getASNumber(Host))
    #Autonomous system number of Website
    Features.extend(tf_idf_predict(tf_idf_Vectorizer, [URL], tf_idf_FileName))
    #Predicted label for the URL using tf_idf and logistic regression
    return Features


if __name__ == "__main__":

    Test_URLs = ['wikipedia.com']
    Train_URLs = ProcessCSV("URL.txt")
    Features = {}
    tf_idf_Vectorizer, tf_idf_FileName = tf_idf_train(ProcessCSV("URL.txt"))
    #Trains the tf_idf and logistic regression model using the current text file
    for URL in Train_URLs[100:150]:
          print(URL)
          Features[URL] = MakeFeatures(URL, tf_idf_Vectorizer, tf_idf_FileName)
          print(Features)
    """
        URL_Details = urlparse(URL[0])
        Host = URL_Details.netloc
        Path = URL_Details.path
        print(len(MakeTokens(Path)))
        print(len(Path))
        print(len(Host))
        #getASNumber(Host)
        """
    """
        Tokens_Main = MakeTokens(URL[0])
        #SecuritySensitive(Tokens)
        print(URL[1], IP_URL(Tokens_Main))
    """
    """
    Train_URLs = ProcessCSV("URL.txt")
    Vectorizer, FileName = tf_idf_train(Train_URLs)
    print(tf_idf_predict(Vectorizer, Test_URLs, FileName))
    for URL in Train_URLs:
        AlexaRanking(URL)
    """
    """
    for URL in Test_URLs:
        URL_Features.append(Tokens(URL))

    df = pd.DataFrame(URL_Features, index = Test_URLs, columns = Labels)
    print(df)
    """
