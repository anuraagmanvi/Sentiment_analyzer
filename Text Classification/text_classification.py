# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:33:16 2019

@author: Anuraag
"""

import numpy
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
	
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

reviews = load_files('txt_sentoken')

x, y = reviews.data, reviews.target

#storing as pickle files
#pickling the data

with open('x.pickle', 'wb') as f:
    pickle.dump(x, f)
    
with open('y.pickle', 'wb') as f:
    pickle.dump(y, f)
    
#unpicklind the data

with open('x.pickle', 'rb') as f:
    f = pickle.load(f)
    

with open('y.pickle', 'rb') as f:
    f = pickle.load(f)
    
#Preprocessing the data and creating the corpus
    
corpus = []
for i in range(0,len(x)):
    review = re.sub(r'\W', ' ', str(x[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

#bag of words vectoriser
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
bow = vect.fit_transform(corpus).toarray()
#convert bag of words model to tfidf model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(bow).toarray()


#tfidf vectoriser directly do the 2 above steps
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvect = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
tfidfvector = tfidfvect.fit_transform(corpus).toarray()


from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(tfidf, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train, sent_train)


sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)

sum = cm[0][0]+cm[1][1]
acc = sum/4

print(acc)

#pickling the classifier
with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)

with open('tfidfmodel.pickle', 'wb') as f:
    pickle.dump(tfidfvect, f)
    
#unpickling the classifier and vectorizer
with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)
    
with open('tfidfmodel.pickle', 'rb') as f:
    tfidf = pickle.load(f)
    
sample = ["You are a nice person. Have a good day."]
sample = tfidf.transform(sample).toarray()
if clf.predict(sample) == 1:
    print("Positive")
else:
    print("Negative")