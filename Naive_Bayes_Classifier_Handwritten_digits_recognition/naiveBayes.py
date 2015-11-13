#!/usr/bin/python

import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import nltk
############################################################################### 

def transfer(fileDj, vocabulary):
	stemmer = nltk.stem.snowball.EnglishStemmer()
	fp = open(fileDj)
	reviews = fp.read().lower()
	reviews = nltk.word_tokenize(reviews)
	stemReviews = [stemmer.stem(word) for word in reviews]
	BOWDj = np.zeros(len(vocabulary))
	for word in stemReviews:
		if word in vocabulary:
			vocIndex = vocabulary.index(word)
			BOWDj[vocIndex] = BOWDj[vocIndex] + 1
		else:
			BOWDj[0] = BOWDj[0] + 1
	fp.close()
	return BOWDj


def loadData(Path):
	vocabulary = ["unknown", "love", "wonderful", "best", "great", "superb", "still", "beautiful", "bad", "worst", "stupid", "waste", "boring", "?", "!"]
	stemmer = nltk.stem.snowball.EnglishStemmer()
	stemVocabulary = [stemmer.stem(word) for word in vocabulary]
	
	dirTrain = os.path.join(Path, "training_set")
	dirTrainPos = os.path.join(dirTrain, "pos")
	dirTrainNeg = os.path.join(dirTrain, "neg")
	Xtrain = []
	ytrain = []
	#print "load data from ", os.path.abspath(dirTrainPos)
	for file in os.listdir(dirTrainPos):
		filePath = os.path.join(dirTrainPos, file)
		Xtrain.append(transfer(filePath, stemVocabulary))
		ytrain.append(1)
	#print "load data from ", os.path.abspath(dirTrainNeg)
	for file in os.listdir(dirTrainNeg):
		filePath = os.path.join(dirTrainNeg, file)
		Xtrain.append(transfer(filePath, stemVocabulary))
		ytrain.append(-1)
	
	dirTest = os.path.join(Path, "test_set")
	dirTestPos = os.path.join(dirTest, "pos")
	dirTestNeg = os.path.join(dirTest, "neg")
	Xtest = []
	ytest = []
	#print "load data from ", os.path.abspath(dirTestPos)
	for file in os.listdir(dirTestPos):
		filePath = os.path.join(dirTestPos, file)
		Xtest.append(transfer(filePath, stemVocabulary))
		ytest.append(1)
	#print "load data from ", os.path.abspath(dirTestNeg)
	for file in os.listdir(dirTestNeg):
		filePath = os.path.join(dirTestNeg, file)
		Xtest.append(transfer(filePath, stemVocabulary))
		ytest.append(-1)
	
	#print type(Xtest), type(ytest)
	Xtrain = np.asarray(Xtrain)
	ytrain = np.asarray(ytrain)
	Xtest = np.asarray(Xtest)
	ytest = np.asarray(ytest)
	#print Xtest.shape, ytest.shape

	return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):
	freqPos = np.sum(Xtrain[ytrain == 1], axis = 0)
	thetaPos = (1 + freqPos) / (len(Xtrain[0]) + Xtrain[ytrain == 1].sum())
	freqNeg = np.sum(Xtrain[ytrain == -1], axis = 0)
	thetaNeg = (1 + freqNeg) / (len(Xtrain[0]) + Xtrain[ytrain == -1].sum())
	return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
	testPos = np.dot(Xtest, np.log(thetaPos).T)
	testNeg = np.dot(Xtest, np.log(thetaNeg).T)
	yPredict = ytest.copy()
	for i in range(len(Xtest)):
		if testPos[i] > testNeg[i]:
			yPredict[i] = 1
		else:
			yPredict[i] = -1
	Accuracy = sum(yPredict == ytest) / float(len(ytest))
	return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
	skMnbc = MultinomialNB()
	skMnbc.fit(Xtrain, ytrain)
	yPredict = skMnbc.predict(Xtest)
	Accuracy = sum(ytest == yPredict)/float(len(ytest))
	return Accuracy


def naiveBayesMulFeature_testDirectOne(path, thetaPos, thetaNeg):
	vocabulary = ["unknown", "love", "wonderful", "best", "great", "superb", "still", "beautiful", "bad", "worst", "stupid", "waste", "boring", "?", "!"]
	stemmer = nltk.stem.snowball.EnglishStemmer()
	stemVocabulary = [stemmer.stem(word) for word in vocabulary]
	bow = transfer(path, stemVocabulary)
	pos = np.inner(bow,np.log(thetaPos))
	neg = np.inner(bow,np.log(thetaNeg))
	if pos > neg :
		yPredict = 1
	else :
		yPredict = -1
	return yPredict


def naiveBayesMulFeature_testDirect(path, thetaPos, thetaNeg):
	
	dirTestPos = os.path.join(path, "pos")
	dirTestNeg = os.path.join(path, "neg")
	yPredict = []
	ytest = []
	#print "load data from ", os.path.abspath(dirTestPos)
	for file in os.listdir(dirTestPos):
		filePath = os.path.join(dirTestPos, file)
		predict = naiveBayesMulFeature_testDirectOne(filePath, thetaPos, thetaNeg)
		yPredict.append(predict)
		ytest.append(1)
		
	#print "load data from ", os.path.abspath(dirTestNeg)
	for file in os.listdir(dirTestNeg):
		filePath = os.path.join(dirTestNeg, file)
		predict = naiveBayesMulFeature_testDirectOne(filePath, thetaPos, thetaNeg)
		yPredict.append(predict)
		ytest.append(-1)

	yPredict = np.asarray(yPredict)
	ytest = np.asarray(ytest)
	Accuracy = sum(yPredict == ytest) / float(len(ytest))
	return yPredict, Accuracy



def naiveBayesBernFeature_train(Xtrain, ytrain):
	posCount = np.zeros(len(vocabulary))
	negCount = np.zeros(len(vocabulary))
	for i in range(len(ytrain)) :
		if ytrain[i] == 1 :
			posCount += np.where(Xtrain[i] > 0, 1, 0)
		else :
			negCount += np.where(Xtrain[i] > 0, 1, 0)
	thetaPosTrue = (posCount + 1) / (sum(y == 1 for y in ytrain) + 2)
	thetaNegTrue = (negCount + 1) / (sum(y == -1 for y in ytrain) + 2)
	return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
	yPredict = []
	testPos = np.dot(Xtest, np.log(thetaPosTrue).T)
	testNeg = np.dot(Xtest, np.log(thetaNegTrue).T)
	yPredict = ytest.copy()
	for i in range(len(ytest)) :
		if testPos[i] >= testNeg[i] :
			yPredict[i] = 1
		else :
			yPredict[i] = -1
	Accuracy = sum(yPredict == ytest) / float(len(ytest))
	return yPredict, Accuracy

'''
Xtrain, Xtest, ytrain, ytest = loadData("data_sets_small")
thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
print thetaPos, thetaNeg
yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg)
print ytest, yPredict, Accuracy
Accuracy = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
print "skMnbc", Accuracy
'''	
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()

    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg
    print "--------------------"

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"
