import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from textblob import TextBlob
import matplotlib.pyplot as plt
import re
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer #word stemmer class
lemma = WordNetLemmatizer()
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pickle
import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
"# -- coding: utf-8 --"


def calctime(a):
    return time.time() - a


positive = 0
negative = 0
compound = 0
vectorizer = pickle.load(open("vector.pickel", "rb"))
w = pickle.load(open('x_train1', 'rb'))
x_train=w['x_train']
vectorizer.fit(x_train)
print(vectorizer)
print(len(vectorizer.get_feature_names()))
filename = 'logR.sav'

# filename='logR.sav'
loaded_model = joblib.load(open(filename, 'rb'))
ps = PorterStemmer()
wnl = WordNetLemmatizer()
count = 0
initime = time.time()
plt.ion()

import test

ckey = 'WnyAgUaacX1YheRSJqwMhhZgR'
csecret = 'LzHg7GuAfJNIsHRpRXEk72TaEjcG5RL9yl85c0rbI1V1pg6rHQ'
atoken = "1125091796046843905-DNeIxEe9RNwlwzZZXwXEW3VJFlv7Az"
asecret = "n3Yc9GzA2Saa6LNPZ5465WdQNj06G6hBrqcWnpwkc4jCb"


class listener(StreamListener):

    def on_data(self, data):
        global initime

        all_data = json.loads(data)
        tweet = all_data["text"]
        # username=all_data["user"]["screen_name"]
        # tweet = " ".join(re.findall("[a-zA-Z]+", tweet))
        # blob = TextBlob(tweet.strip())
        global ps
        global wnl

        sequencePattern = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"
        tweets = " ".join(filter(lambda x: x[0] != '@', tweet.split()))
        tweets = re.sub(r'([^a-zA-Z0-9])', ' ', tweets)
        tweets = re.sub(r'[^\w\s]', '', tweets)
        #     tweets=re.sub('[\s][a-z]{1,3}[\s]',' ',tweets)
        #     tweets=re.sub('^[a-z]{1,3}[\s]',' ',tweets)
        tweets = re.sub(r'[0-9_]', '', tweets)
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweets)
        tweets = tweets.lower()
        tweets = re.sub(sequencePattern, seqReplacePattern, tweets)
        #     print(tweets)
        tweets = tweets.split()
        tweets = [word for word in tweets if not word in set(stopwords.words('english'))]
        tweets = [lemma.lemmatize(word) for word in tweets]

        tweets = " ".join(tweets)

        #     print(tweets[0])
        #     tweets=tweets.split(" ")

        tweets = word_tokenize(tweets)
            # print(tweets)
        t = []
        for j in tweets:
            #
            t.append(ps.stem(j))
            #            t.append(wnl.lemmatize(j))
            t.append(" ")

        tweets = " ".join(t)
        #     tweets = tweets.split()
        tweets = tweets.replace('ing', '')
        tweets = tweets.replace('pic', '')
        tweets = tweets.replace('com', '')

        # l = vectorizer.transform(tweets).toarray()
        # tweets='fuck corona'
        blob = TextBlob(tweets.strip())
        k = pd.Series(tweets)
        # print(k)

        l = vectorizer.transform(k).toarray()
        # print(l)
        m=loaded_model.predict(l)

        global positive
        global negative
        global compound
        global count

        count = count + 1
        senti = 0
        for sen in blob.sentences:
            senti = senti + sen.sentiment.polarity
            if sen.sentiment.polarity >= 0:
                positive = positive + sen.sentiment.polarity
            else:
                negative = negative + sen.sentiment.polarity
        compound = compound + senti
        print
        count
        print
        tweet.strip()
        print
        senti
        print
        t
        print
        str(positive) + ' ' + str(negative) + ' ' + str(compound)

        plt.axis([0, 70, -20, 20])
        plt.xlabel('Time')
        plt.ylabel('Sentiment')
        plt.plot([t], [positive], 'go', [t], [negative], 'ro', [t], [compound], 'bo')
        plt.show()
        plt.pause(0.1)
        if count == 200:
            return False
        else:
            return True

    def on_error(self, status):
        print
        status


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener(count))
twitterStream.filter(track=["Donald Trump"])
