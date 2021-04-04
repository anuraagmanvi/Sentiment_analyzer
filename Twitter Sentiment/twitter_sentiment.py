# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 23:47:11 2019

@author: Anuraag
"""

import tweepy
import re
import pickle

from tweepy import OAuthHandler

consumer_key='2OhsuxRHdhL4DyVqYFz9dpX2I'
consumer_sec='j4lukJteYZxYtJjPNJuUC9tAPj8P5uoblrVUINZOwOg5s7Fa5P'
access_tok='1169941496688070656-Tl5eypq5TCNph0BfhqgjPRTAVzkERC'
access_sec='e4fo5bB5HFGjNsFOqjqZzRt6CM7ZYIB7TD8b63KwzEC2U'

auth = OAuthHandler(consumer_key, consumer_sec)
auth.set_access_token(access_tok, access_sec)
topic = input("What do you want to analyse on twiter today?")
args = [topic]
api = tweepy.API(auth, timeout=10)

tweets_list = []

query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search, q=query+" -filter:retweets", lang='en', result_type='recent').items(500):
        tweets_list.append(status.text)
        
with open('tfidfmodel.pickle', 'rb') as f:
    vect = pickle.load(f)
    
with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)

total_pos = 0
total_neg = 0

for tweet in tweets_list:
    tweet = re.sub(r"^https://t.co/[a-zA-Z-0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z-0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z-0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"where's", "where is", tweet)
    tweet = re.sub(r"there's", "there is", tweet)
    tweet = re.sub(r"what's", "what is", tweet)
    tweet = re.sub(r"who's", "who is", tweet)
    tweet = re.sub(r"that's", "that is", tweet)
    tweet = re.sub(r"it's", "it is", tweet)
    tweet = re.sub(r"I'm", "I am", tweet)
    tweet = re.sub(r"she's", "she is", tweet)
    tweet = re.sub(r"don't", "do not", tweet)
    tweet = re.sub(r"he's", "he is", tweet)
    tweet = re.sub(r"they're", "they are", tweet)
    tweet = re.sub(r"who're", "who are", tweet)
    tweet = re.sub(r"would'nt", "would not", tweet)
    tweet = re.sub(r"should'nt", "should not", tweet)
    tweet = re.sub(r"can't", "can not", tweet)
    tweet = re.sub(r"could'nt", "could not", tweet)
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"\W", " ", tweet)
    tweet = re.sub(r"\d", " ", tweet)
    #tweet = re.sub(r"\s+[a-z]\s", " ", tweet)
    #tweet = re.sub(r"\s+[a-z]$", " ", tweet)
    #tweet = re.sub(r"^[a-z]\s+", " ", tweet)
    tweet = re.sub(r"\s+", " ", tweet)
    sent = clf.predict(vect.transform([tweet]).toarray())
    if sent[0] == 1:
        total_pos +=1
    else:
        total_neg +=1

import matplotlib.pyplot as plt
import numpy as np
objects = ['Positive' + str(total_pos), 'Negative' + str(total_neg)]
y_pos = np.arange(len(objects))

plt.bar(y_pos, [total_pos, total_neg], alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Numbers')
plt.title('Sentiment on Tweets')
plt.show()
