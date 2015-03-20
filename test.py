#from __future__ import division
import re
import csv
import nltk
import pickle
import gettweets
#initialize stopWords
stopWords = []


#For substitution and cleaning tweets:
def processTweet(tweet):
	#Remove the keyword RT
	tweet = re.sub('RT','',tweet)
	#converting everything in lower case.
	tweet = tweet.lower()
	#Removing the URL's in the tweet, if any.
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
	#Removing Smilies/Emoticons.
	tweet = re.sub('([\:\;][\)\|\\\/dDOoPp\(\'\"][\(\)DOo]?)','',tweet)
	#Removing the @ sign. Instead just putting the term.
	tweet = re.sub('@[^\s]+','',tweet)
	#Removing the # and _ term.
	tweet = re.sub('[#_]','',tweet)
	#Converting the ..... in . and --- in -
	tweet = re.sub('[.]+','.',tweet)
	#Converting --- in -
	tweet = re.sub('[-]+','-',tweet)
	#Removing more than 1 whitespace in 1.
	tweet = re.sub('[\s]+',' ',tweet)
	#Passing value..
	return tweet

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    #stopWords.append('AT_USER')
    #stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

#Start GetFeatureList:
def getFeatureList(FeatureListFile):
	featureList = []
	readfile = open(FeatureListFile,'rb')
	word = readfile.readline()
	while word:
		#print(word)
		featureList.append(str(word))
		word = readfile.readline()
	return featureList
#end



featureList = []
#Read the tweets one by one and process it
stopWords = getStopWordList('stopwords.txt')
#Using feauture list from text file.
#This gets the next-line whitespace. Using REGEX to remove it.
featureListtemp = getFeatureList('featureList.txt')
#featureList = []
for i in featureListtemp:
	i = re.sub('[\s]','',i)
	featureList.append(i)	
#Cleaning Variable. Freeing memory.
featureListtemp = 0	
#Removing the duplicates:
featureList = list(set(featureList))
#print(featureList)	


#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


# Train the classifier- Naive Bayes..
#NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
#print("hello1")
#f = open('NBClassifier.pickle','wb')
#pickle.dump(NBClassifier,f)
#f.close()

#TO LOAD THE PICKLE CLASSIFIERS..
def loadeverything(searchTerm):
	#featureList =  initialize()
	f = open('NBClassifier.pickle')
	NBClassifier = pickle.load(f)
	f.close()	
	
	#Count to check Accuracy:
	count = int(0)
	#accurate = int(0)
	#use tuple
	tweet_set = []
	#remove text file stuff..
	td = gettweets.TwitterData()
	ALL_TWEETS = td.getData(searchTerm)	

	for singletweet in ALL_TWEETS:
		count = count + 1
		testTweet = singletweet
		processedTestTweet = processTweet(testTweet)
		test1 = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet,stopWords)))
		tweet_set.append((count,singletweet,test1))
	return tweet_set
#END

#y = "blackberry"
#x = loadeverything(y)
#for i in range(0,len(x)):
#	print(x[i][0])
#	print(x[i][1])
#	print(x[i][2])
#	print("\n")
