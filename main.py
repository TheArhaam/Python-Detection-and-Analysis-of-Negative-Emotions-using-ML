# PROJECT: DETECTION AND ANALYSIS OF NEGATIVE EMOTIONS ON SOCIAL MEDIA USING MACHINE LEARNING
# Name:         Arhaam Patvi
# PRN:          17070124501
# Institute:    Symbiosis Institute of Technology
# Branch:       IT
# Batch:        2016-20


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
     
        self.root.title('Detection and Analysis of Negative Emotions using ML')

        self.twitterULabel = Label(root,text="Enter Twitter Username: ")
        self.twitterULabel.config(font=('','15'))

        strvar = StringVar(root)
        self.twitterUEntry = Entry(root,width=50,textvariable=strvar)
        self.twitterUEntry.config(font=('','12'))

        self.submitBttn = tkinter.Button(root, text="SUBMIT", padx=10,pady=5, command=self.DownloadData)

        self.twitterULabel.grid(row=0,column=0,padx=5,pady=5)
        self.twitterUEntry.grid(row=1,column=0,padx=10,pady=5)
        self.submitBttn.grid(row=2,column=0,padx=5,pady=5);

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
        self.tweets = api.user_timeline(screen_name=self.twitterUname, count=self.NoOfTerms)

        # Open/create a file to append data to (Delete if existing)
        # if(os.path.exists('result.csv')):
        #     os.remove('result.csv')
        # time.sleep(4)
        # csvFile = open('result.csv', 'a')
        # Use csv writer
        # csvWriter = csv.writer(csvFile)
        # Write to csv and close csv file
        # csvWriter.writerow(self.tweetText)
        # csvFile.close()

        sa.NLPAnalysis()

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
        # iterating through tweets fetched
        for tweet in self.tweets:
            #Append to temp so that we can store in csv later. I use encode UTF-8
            self.tweetText.append(self.cleanTweet(tweet.text).encode('utf-8'))
            # print (tweet.text.translate(non_bmp_map))    #print tweet's text
            analysis = TextBlob(tweet.text)
            # print(analysis.sentiment)  # print tweet's polarity
            self.polarity += analysis.sentiment.polarity  # adding up polarities to find the average later

            if (analysis.sentiment.polarity == 0):  # adding reaction of how people are reacting to find average later
                self.neutral += 1
            elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):
                self.wpositive += 1
                # positive += 1
            elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
                self.positive += 1
            elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
                self.spositive += 1
                # positive += 1
            elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
                self.wnegative += 1
                # negative += 1
            elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
                self.negative += 1
            elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
                self.snegative += 1
                # negative += 1
        
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
        sa.plotPieChart()

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
        # labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]','Negative [' + str(negative) + '%]']
        sizes = [positive, wpositive, spositive, neutral, negative, wnegative, snegative]
        # sizes = [positive, neutral, negative]
        colors = ['yellowgreen','lightgreen','darkgreen', 'gold', 'red','lightsalmon','darkred']
        
        fig = matplotlib.figure.Figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        patches, texts = ax.pie(sizes, colors=colors, startangle=90)
        ax.legend(patches, labels, loc="best")
        # ax1.title.set_text('First Plot')
        ax.title.set_text('ANALYSING TWEETS OF USER: '+self.twitterUname+' TWEET COUNT: '+str(self.NoOfTerms))
        # ax.axis('equal')
        # ax.tight_layout()
        canvas = FigureCanvasTkAgg(fig,master=self.root)
        canvas.get_tk_widget().grid(row=3,column=0,padx=5,pady=5)



if __name__== "__main__":
    root = tkinter.Tk()
    sa = SentimentAnalysisNLP(root)
    root.mainloop()
