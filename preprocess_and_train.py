import os
import time
import itertools
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
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

def plot_roc_curve(model, X_test, Y_test):
    probas_ = model.predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC Curve (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.set_tight_layout(True)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

start_time = time.time()
input_file_name = 'reviews.csv'
preprocess_flag = 0 #if 1, pre-process csv file from scratch and dump train and test data- if 0, load the train and test data already dumped
X_train, Y_train, X_test, Y_test = read_data(input_file_name, preprocess_flag)

#print(len(X_train))
#print(len(X_test))
if not preprocess_flag:
    bow_transformer = joblib.load('FeatTransformer.pkl')
    X_train = joblib.load('TrainFeatures.pkl')
    X_test = joblib.load('TestFeatures.pkl')
else:
    bow_transformer = CountVectorizer(analyzer=format_sentence).fit(X_train)
    X_train = bow_transformer.transform(X_train)
    X_test = bow_transformer.transform(X_test)
    joblib.dump(bow_transformer, 'FeatTransformer.pkl')
    joblib.dump(X_train, 'TrainFeatures.pkl')
    joblib.dump(X_test, 'TestFeatures.pkl')

#train decision tree classifier
dt_flag = 0 #if 1, train model from scratch and dump - if 0, load dumped model
dt = DecisionTreeClassifier()
if dt_flag:
    dt_clf = dt.fit(X_train, Y_train)
    joblib.dump(dt_clf, 'DTmodel.pkl') 
else:
    dt_clf = joblib.load('DTmodel.pkl') 
#test dt classifier
preds = dt_clf.predict(X_test)
cm = confusion_matrix(Y_test, preds)
print(cm)
print('\n')
print(classification_report(Y_test, preds))
#plot_roc_curve(dt_clf,X_test,Y_test)
plt.figure()
plot_confusion_matrix(cm, classes=['negative', 'positive'], normalize=True, title='Normalized confusion matrix - Decision Tree')
plt.show()

#train naive bayes classifier
nb_flag = 0 #if 1, train model from scratch and dump - if 0, load dumped model
nb = MultinomialNB()
if nb_flag:
    nb_clf = nb.fit(X_train, Y_train)
    joblib.dump(nb_clf, 'NBmodel.pkl')
else:
    nb_clf = joblib.load('NBmodel.pkl')
#test nb classifier
preds = nb_clf.predict(X_test)
cm = confusion_matrix(Y_test, preds)
print(cm)
print('\n')
print(classification_report(Y_test, preds))
#plot_roc_curve(nb_clf,X_test,Y_test)
plt.figure()
plot_confusion_matrix(cm, classes=['negative', 'positive'], normalize=True, title='Normalized confusion matrix - Naive Bayes')
plt.show()

#train neural network classifier 
nn_flag = 0
nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,5), random_state=1)
if nn_flag:
    nn_clf = nn.fit(X_train, Y_train)
    joblib.dump(nn_clf, 'NNmodel.pkl')
else:
    nn_clf = joblib.load('NNmodel.pkl')
preds = nn_clf.predict(X_test)
cm = confusion_matrix(Y_test, preds)
print(cm)
print('\n')
print(classification_report(Y_test, preds))
#plot_roc_curve(nb_clf,X_test,Y_test)
plt.figure()
plot_confusion_matrix(cm, classes=['negative', 'positive'], normalize=True, title='Normalized confusion matrix - Neural Network')
plt.show()

'''
#perform sentiment analysis on a new review instance
with codecs.open(sys.argv[2],'r', encoding='utf8') as text_data:
    x_test = bow_transformer.transform(text_data)
    y_test = preds = nb_clf.predict(x_test)
with open('output.txt', 'w') as f:
    f.write('%d' % y_test)
    f.close()
print 'runtime for handling the test case: %s' % (time.time() - start_time)
'''
