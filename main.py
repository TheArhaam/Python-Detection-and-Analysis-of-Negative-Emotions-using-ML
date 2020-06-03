# PROJECT: DETECTION AND ANALYSIS OF NEGATIVE EMOTIONS ON SOCIAL MEDIA USING MACHINE LEARNING
# Name:         Arhaam Patvi
# PRN:          17070124501
# Institute:    Symbiosis Institute of Technology
# Branch:       IT
# Batch:        2016-20

#region IMPORTS
import sys,tweepy,csv,re
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import time
import tkinter
from tkinter import *
import random
import sklearn.metrics
import pandas as pd
import keras
import keras.layers
import keras.optimizers
import keras.utils
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Dropout, SimpleRNN, Conv2D, Conv1D, Flatten, MaxPooling2D, MaxPooling1D, Embedding, LSTM, GRU, Bidirectional, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import queue
from tkinter.ttk import *
import threading
#endregion

# TO-DO
# TRY TO ANALYZE ML OUTPUT

# FIRST APPROACH
# SENTIMENT ANALYSIS USING NATURAL LANGUAGE PROCESSING
class SentimentAnalysisNLP:

    def __init__(self,root):
        self.tweets = []
        self.tweetText = []
        self.polarity = int()
        self.positive = int()
        self.wpositive = int()
        self.spositive = int()
        self.negative = int()
        self.wnegative = int()
        self.snegative = int()
        self.neutral = int()
        self.NoOfTerms = int()
        self.twitterUname = str()
        self.root = root
        self.saml = SentimentAnalysisML()
        
        self.root.title('Detection and Analysis of Negative Emotions using ML')

        self.twitterULabel = Label(root,text="Enter Twitter Username: ")
        self.twitterULabel.config(font=('','15'))
        self.successLabel = Label(root,text="SUCCESS")

        strvar = StringVar(root)
        self.twitterUEntry = Entry(root,width=50,textvariable=strvar)
        self.twitterUEntry.config(font=('','12'))

        self.submitBttn = tkinter.Button(root, text="SUBMIT", padx=10, pady=5, command=self.DownloadData)
        self.MLBttn = tkinter.Button(root, text="NEXT (ML Approach)", padx=10, pady=5, command=self.saml.start)

        self.twitterULabel.grid(row=0,column=0,padx=5,pady=5)
        self.twitterUEntry.grid(row=1,column=0,padx=10,pady=5, rowspan=2)
        self.submitBttn.grid(row=3,column=0,padx=5,pady=5, rowspan=2);

    def DownloadData(self):
        # authenticating
        consumerKey = 'ggHLlqgbMR9izg4JJPrTDyvc1'
        consumerSecret = 'mDs1rEuxLIoysAwYpPMDe6AbebEZI1FDpBomjRIBJoSairU0KN'
        accessToken = '1220759752948404224-CFjJMkM5nzpa9UqV8ceNMSnMB7kEML'
        accessTokenSecret = 'Zaj8nZjYDNUrCjxe4CVJxzTygbQR5nPNvVhe3j4pr9sFX'
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)

        self.twitterUname = self.twitterUEntry.get()

        # GETTING NUMBER OF TWEETS
        self.NoOfTerms = api.get_user(self.twitterUname).statuses_count
        # Limit of 200 tweets
        if(self.NoOfTerms>200):
            self.NoOfTerms = 190

        # searching for tweets
        self.tweets = api.user_timeline(screen_name=self.twitterUname, count=self.NoOfTerms,tweet_mode = 'extended')

        self.NLPAnalysis()

    def NLPAnalysis(self):
        # Initializing variables
        self.polarity = 0
        self.positive = 0
        self.wpositive = 0
        self.spositive = 0
        self.negative = 0
        self.wnegative = 0
        self.snegative = 0
        self.neutral = 0

        # Opening file to write tweets
        testFile = open('2018-E-c-En-test.txt','w',encoding='utf-8')
        testFile.truncate()
        testFile.write('ID\tTweet\tanger\tanticipation\tdisgust\tfear\tjoy\tlove\toptimism\tpessimism\tsadness\tsurprise\ttrust')
        # iterating through tweets fetched
        for tweet in self.tweets:
            analysis = TextBlob(tweet.full_text)
            self.polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            if (analysis.sentiment.polarity == 0):  # adding reaction of how people are reacting to find average later
                self.neutral += 1                
            elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):
                self.wpositive += 1
            elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                self.positive += 1
            elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
                self.spositive += 1
            elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
                self.wnegative += 1
                testFile.write('\n'+self.getTweetId(tweet)+'\t'+self.cleanTweet(self.getTweetText(tweet))+'\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE')
            elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                self.negative += 1
                testFile.write('\n'+self.getTweetId(tweet)+'\t'+self.cleanTweet(self.getTweetText(tweet))+'\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE')
            elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
                self.snegative += 1
                testFile.write('\n'+self.getTweetId(tweet)+'\t'+self.cleanTweet(self.getTweetText(tweet))+'\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE\tNONE')
        testFile.close()


        # finding average of how people are reacting
        self.positive = self.percentage(self.positive, self.NoOfTerms)
        self.wpositive = self.percentage(self.wpositive, self.NoOfTerms)
        self.spositive = self.percentage(self.spositive, self.NoOfTerms)
        self.negative = self.percentage(self.negative, self.NoOfTerms)
        self.wnegative = self.percentage(self.wnegative, self.NoOfTerms)
        self.snegative = self.percentage(self.snegative, self.NoOfTerms)
        self.neutral = self.percentage(self.neutral, self.NoOfTerms)

        # finding average reaction
        self.polarity = self.polarity / self.NoOfTerms
        self.plotPieChart()

    def getTweetId(self,tweet):
        return str(tweet.created_at.year)+str(tweet.created_at.month)+str(tweet.created_at.day)+str(tweet.created_at.hour)

    def getTweetText(self, tweet):
        return tweet.full_text

    def cleanTweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())

    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    def plotPieChart(self):
        polarity = self.polarity
        positive = self.positive
        wpositive = self.wpositive
        spositive = self.spositive
        negative = self.negative
        wnegative = self.wnegative
        snegative = self.snegative
        neutral = self.neutral

        labels = ['Positive [' + str(positive) + '%]', 'Weakly Positive [' + str(wpositive) + '%]','Strongly Positive [' + str(spositive) + '%]', 'Neutral [' + str(neutral) + '%]',
                  'Negative [' + str(negative) + '%]', 'Weakly Negative [' + str(wnegative) + '%]', 'Strongly Negative [' + str(snegative) + '%]']
        sizes = [positive, wpositive, spositive, neutral, negative, wnegative, snegative]
        colors = ['yellowgreen','lightgreen','darkgreen', 'gold', 'red','lightsalmon','darkred']
        
        fig = matplotlib.figure.Figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        patches, texts = ax.pie(sizes, colors=colors, startangle=90)
        ax.legend(patches, labels, loc="best")
        ax.title.set_text('USER: '+self.twitterUname+' TWEETS: '+str(self.NoOfTerms))
        canvas = FigureCanvasTkAgg(fig,master=self.root)
        canvas.get_tk_widget().grid(row=5,column=0,padx=5,pady=5,rowspan=23)
        self.MLBttn.grid(row=29,column=0,padx=5,pady=5, rowspan=2)

    def samlComplete(self):
        self.successLabel.grid(row=5,column=0,padx=5,pady=5)
        self.ofile = open('E-C_en_pred.txt','r',encoding='utf-8')
        self.rows = sum(1 for line in self.ofile)
        self.ofile.close()
        self.ofile = open('E-C_en_pred.txt','r',encoding='utf-8')

        # TITLES
        self.titles = self.ofile.readline().split('\t')
        self.l1 = Label(self.root,text=self.titles[1], font=('bold'))
        self.l2 = Label(self.root,text=self.titles[2], font=('bold'))
        self.l3 = Label(self.root,text=self.titles[4], font=('bold'))
        self.l4 = Label(self.root,text=self.titles[5], font=('bold'))
        self.l5 = Label(self.root,text=self.titles[9], font=('bold'))
        self.l6 = Label(self.root,text=self.titles[10], font=('bold'))
        self.l1.grid(row=0,column=1)
        self.l2.grid(row=0,column=2)
        self.l3.grid(row=0,column=3)
        self.l4.grid(row=0,column=4)
        self.l5.grid(row=0,column=5)
        self.l6.grid(row=0,column=6)


        for i in range(1,self.rows):
            line = self.ofile.readline().split('\t')
            self.e1 = Entry(self.root,width=100)
            self.e2 = Entry(self.root,width=15)
            self.e3 = Entry(self.root,width=15)
            self.e4 = Entry(self.root,width=15)
            self.e5 = Entry(self.root,width=15)
            self.e6 = Entry(self.root,width=15)  

            self.e1.grid(row=i,column= 1)
            self.e2.grid(row=i,column= 2)
            self.e3.grid(row=i,column= 3)
            self.e4.grid(row=i,column= 4)
            self.e5.grid(row=i,column= 5)
            self.e6.grid(row=i,column= 6)
            
            self.e1.insert(END,line[1]) 
            self.e2.insert(END,line[2]) 
            self.e3.insert(END,line[4]) 
            self.e4.insert(END,line[5]) 
            self.e5.insert(END,line[9]) 
            self.e6.insert(END,line[10]) 

# SECOND APPROACH
# SENTIMENT ANALYSIS USING MACHINE LEARNING
class SentimentAnalysisML():

    def start(self):
        emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
        emotion_to_int = {"0": 0, "1": 1, "NONE": -1}

        read_csv_kwargs = dict(sep="\t",converters={e: emotion_to_int.get for e in emotions})
        train_data = pd.read_csv("2018-E-c-En-train.txt", **read_csv_kwargs)
        dev_data = pd.read_csv("2018-E-c-En-dev.txt", **read_csv_kwargs)
        test_data = pd.read_csv("2018-E-c-En-test.txt", **read_csv_kwargs)

        t = Tokenizer(lower=True)
        t.fit_on_texts(train_data['Tweet'])
        word2id = t.word_index
        id2word = {word2id[i]: i for i in word2id}
        vocab_size = len(t.word_index) + 1
        encoded_docs_train = t.texts_to_sequences(train_data['Tweet'])
        encoded_docs_dev = t.texts_to_sequences(dev_data['Tweet'])
        encoded_docs_test = t.texts_to_sequences(test_data['Tweet'])

        # this is 36 on the train data, used to determine padding
        max_doc_train = 0
        for i in encoded_docs_train:
            if len(i) > max_doc_train:
                max_doc_train = len(i)

        x_train = pad_sequences(encoded_docs_train, maxlen=max_doc_train, padding='post')
        y_train = np.zeros((len(encoded_docs_train), 11))
        for row in range(len(train_data)):
            y_train[row] = train_data.iloc[row][emotions]


        x_dev = pad_sequences(encoded_docs_dev, maxlen=max_doc_train, padding='post')
        y_dev = np.zeros((len(encoded_docs_dev), 11))
        for row in range(len(dev_data)):
            y_dev[row] = dev_data.iloc[row][emotions]

        x_test = pad_sequences(encoded_docs_test, maxlen=max_doc_train, padding='post')

        # takes about 5 mins and does NOT load everything, because there are words
        # in word2id that do not appear in the GloVe vectors (16k vs. 11k words)
        emb_index = {}
        with open('glove.840B.300d.txt', 'r',encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in word2id:
                    emb_index[word] = values[1:]

        # this matrix will keep the zeros for the indices that do not appear in GloVe vectors
        emb_matrix = np.random.uniform(low=-0.0001, high=0.0001, size=(vocab_size, 300))

        for word in word2id:
            # there are about 10 words that have length 301 due to edge cases in line.split()
            if word in emb_index and len(emb_index[word]) != 301:
                emb_matrix[word2id[word]] = emb_index[word]

        #RNN
        network = Sequential()
        network.add(Embedding(vocab_size, 300, weights=[emb_matrix], input_length=max_doc_train, trainable=False, mask_zero=True))
        # network.add(SimpleRNN(units=128, activation='tanh'))
        network.add(LSTM(units=128, activation='tanh'))
        network.add(BatchNormalization(axis=-1))
        network.add(Dense(units=50, activation='tanh'))
        network.add(Dropout(rate=0.3))
        network.add(Dense(units=25, activation='tanh'))
        network.add(Dropout(rate=0.3))
        network.add(Dense(units=11, activation='sigmoid'))
        network.compile(loss='binary_crossentropy', optimizer='adam')
        network.fit(x=x_train, y=y_train, batch_size=16, epochs=30, verbose=1)

        vec = np.vectorize(int)

        pred_train = vec(np.rint(network.predict(x_train)))
        acc_train = sklearn.metrics.jaccard_similarity_score(pred_train, y_train)

        pred_dev = vec(np.rint(network.predict(x_dev)))
        acc_dev = sklearn.metrics.jaccard_similarity_score(pred_dev, y_dev)

        pred_test = vec(np.rint(network.predict(x_test)))

        print("train accuracy: %s" % acc_train)
        print("dev accuracy: %s" % acc_dev)

        dev_predictions = dev_data.copy()
        dev_predictions[emotions] = pred_dev

        test_predictions = test_data.copy()
        test_predictions[emotions] = pred_test

        #saves predictions and prints out multi-label accuracy
        test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
        print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        dev_data[emotions], dev_predictions[emotions])))
        
        sanlp.samlComplete()

if __name__== "__main__":
    root = tkinter.Tk()
    sanlp = SentimentAnalysisNLP(root)
    root.mainloop()
