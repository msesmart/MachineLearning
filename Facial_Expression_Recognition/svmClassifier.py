#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__='Chengjun Yuan @UVa'

import sys,os
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

def loadDataSet(fileName):  
	absPath=os.path.abspath(fileName)
	data=np.loadtxt(absPath,delimiter=',',dtype ='S')
	data=data[:,1:45]
	return data.astype(int)

def simpleAnalyze(data):
	for i in range(8):
		mask=data[:,43]==i
		index=np.arange(1,44)
		index=np.array([index]).T
		maskData=np.transpose(data[mask,0:43])
		maskData=np.sum(maskData,axis=1)
		maskData=maskData.astype(float)/mask.sum()
		maskData=np.array([maskData]).T
		print index.ndim, maskData.ndim
		maskData=np.append(index,maskData,axis=1)
		print i,'---',mask.sum()
		fileName='emotion_'+str(i)+'.txt'
		np.savetxt(fileName,maskData,delimiter=' ')
	return
	
def getAccuracy(predictResult,actualResult):
    count=0.0
    for i in range(len(predictResult)):
        if(predictResult[i]-actualResult[i]==0):
            count=count+1
    return count/float(len(predictResult))

def processDataSet(train,test):
	print "reading train set"
	trainSet,trainResult=preprocessDataSet(train)
	print "reading test set"
	testSet,testResult=preprocessDataSet(test)
	C_=10.0
	gamma_=0.2
    #'linear','poly','rbf'
	print "start svm "
	svmPredict=SVC(C=C_,gamma=gamma_,kernel='rbf')
	svmPredict.fit(trainSet,trainResult)
	print "start svm prediction"
	predictResult=svmPredict.predict(trainSet)
	trainAccuracy=getAccuracy(predictResult,trainResult)
	print "trainAccuracy",trainAccuracy
	predictResult=svmPredict.predict(testSet)
	testAccuracy=getAccuracy(predictResult,testResult)
	print "testAccuracy",testAccuracy
	stringTable=np.array(["<=50K",">50K"])
	predictions=stringTable[predictResult]
	return predictions

def main(argv):
	data=loadDataSet('CKPlus_AUEmotion_DataSet.csv')
	#print data
	simpleAnalyze(data)
	
if __name__ == "__main__":
	main(sys.argv[1:])