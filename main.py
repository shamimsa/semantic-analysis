import os
import time
import numpy as np
import sklearn
import nltk
import pandas as pd
import sys
import codecs
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier
from nltk.corpus import stopwords
#from nltk.sentiment import SentimentAnalyzer
#import string 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib

stop_words = stopwords.words('english')
#in_file = open(sys.argv[1],"r")
#print in_file
#data = in_file.readlines()
 
def read_data(input_file_name, preprocess_flag):
    if preprocess_flag:
        X_train = [] #train features
        X_test = [] #test features
        Y_train = [] #train labels
        Y_test = [] #test labels
        ct = 0 #line count for train-test split
        with codecs.open(input_file_name,'r', encoding='utf8') as in_file:
            lines = list(in_file)
            for line in lines:
                ct += 1
                sentiment_true = line[:8]
                if sentiment_true == 'negative':
                    sentiment_gt = 0
                else:
                    sentiment_gt = 1 
                review = line[9:]
                content = review[:-2]
                if ct % 5 == 0:
                    X_test.append(content)
                    Y_test.append(sentiment_gt)
                else:
                    X_train.append(content)
                    Y_train.append(sentiment_gt)
            joblib.dump(X_train, 'TrainFeat.pkl')
            joblib.dump(X_test, 'TestFeat.pkl')
            joblib.dump(Y_train, 'TrainLabels.pkl')
            joblib.dump(Y_test, 'TestLabels.pkl')
    else:
        X_train = joblib.load('TrainFeat.pkl')    
        Y_train = joblib.load('TrainLabels.pkl')    
        X_test = joblib.load('TestFeat.pkl')    
        Y_test = joblib.load('TestLabels.pkl')
    return X_train, Y_train, X_test, Y_test    

def format_sentence(sent):
    #return({word: True for word in nltk.word_tokenize(sent.encode('utf-8'))})
    tokens = nltk.word_tokenize(sent)
    tokens_punc_rmvd = remove_punctuation(tokens)
    tokens_no_stpwrd = remove_stop_words(tokens_punc_rmvd)
    #token_dict = {word: True for word in tokens_no_stpwrd}
    tokens_cleaned = tokens_no_stpwrd
    return tokens_cleaned

def remove_punctuation(words): #remove punctuation and convert to lower letters
    return [word for word in words if word.isalpha()]

def remove_stop_words(words):
    return [w for w in words if not w.lower() in stop_words]

start_time = time.time()
bow_transformer = joblib.load('FeatTransformer.pkl')
#clf = joblib.load('DTmodel.pkl') 
#clf = joblib.load('NBmodel.pkl') 
clf = joblib.load('NNmodel.pkl') 

#perform sentiment analysis on a new review instance
with codecs.open(sys.argv[2],'r', encoding='utf8') as text_data:
    x_test = bow_transformer.transform(text_data)
    y_test = clf.predict(x_test)
with open('output.txt', 'w') as f:
    f.write('%d' % y_test)
    f.close()
print 'runtime for handling the test case: %s' % (time.time() - start_time)
