#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__='Chengjun Yuan @UVa'

import sys,os
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn import cross_validation

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
	C_ = 10.0
	gamma_ = 0.2
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

def svm(xTrain, yTrain, xTest, yTest, kernel, C, degree, gamma):
	if gamma == 'auto' :
		svm = SVC(kernel = kernel, C = C, degree = degree)
	else :
		svm = SVC(kernel = kernel, C = C, degree = degree, gamma = gamma)
	svm.fit(xTrain, yTrain)
	yPredict = svm.predict(xTest)
	testAccuracy = svm.score(xTest, yTest)
	accurateNum = sum(yPredict == yTest)
	#print 'testAccuracy: ' , testAccuracy
	return testAccuracy, accurateNum	

def LOO_Svm(data, kernel, C, degree, gamma):
	loo = cross_validation.LeaveOneOut(len(data))
	totalAccurateNum = 0
	for trainIndex, testIndex in loo:
		trainX, trainY = data[trainIndex, 0:43], data[trainIndex, 43]
		testX, testY = data[testIndex, 0:43], data[testIndex, 43]
		testAccuracy, accurateNum = svm(trainX, trainY, testX, testY, kernel, C, degree, gamma)
		totalAccurateNum +=accurateNum
	accuracy = float(totalAccurateNum)/len(data)	
	return accuracy


def plotLogContour(x, y, z, xName, yName, zName):
	fig, ax = plt.subplots()
	if yName == 'gamma' :
		ax.set_yscale('log')
	if xName == 'C' :
		ax.set_xscale('log')
	CS = plt.contourf(x, y, z.T)
	plt.contour(x, y, z.T, 12)
	cb = plt.colorbar(CS, orientation = 'vertical')
	cb.set_label(zName)
	plt.xlabel(xName)
	plt.ylabel(yName)
	plt.show()
	return 
	
def tuneSvm(data):
	C_Gamma = []
	C_Degree = []
	cRange = [0.01, 0.025, 0.1, 0.25, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 16, 25, 100, 250, 500, 1000]
	gammaRange = [0.00001, 0.0001, 0.001, 0.002, 0.004, 0.01, 0.1, 0.25, 1.0, 2.5, 10.0]
	degreeRange = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	for C in cRange:
		print "C=", C
		gamma = []
		for Gamma in gammaRange:
			gamma.append(LOO_Svm(data, 'rbf', C, 1, Gamma)) 
		gamma = np.asarray(gamma)
		C_Gamma.append(gamma)
		
		degree = []
		for Degree in degreeRange:
			degree.append(LOO_Svm(data, 'poly', C, Degree, 'auto'))
		degree = np.asarray(degree)
		C_Degree.append(degree)

	C_Gamma = np.asarray(C_Gamma)
	C_Degree = np.asarray(C_Degree)
	np.savetxt('C_Gamma.txt', C_Gamma)
	np.savetxt('C_Degree.txt', C_Degree)
	plotLogContour(cRange, gammaRange, C_Gamma, 'C', 'gamma', 'accuracy')
	plotLogContour(cRange, degreeRange, C_Degree, 'C', 'degree', 'accuracy')
	return 
	
def threeLevelSvm(xTrain, yTrain, xTest, yTest):
	level_1 = [1, 2, 4]
	level_2 = [5]
	level_3 = [3, 7, 6]
	fake_1 = 5
	fake_2 = 4
	column_mouth = range(9, 34);
	column_eyes = [0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42]
	column_nose = [8, 35, 36, 37, 38, 39]
	svm_1 = SVC(kernel = "rbf", C = 4.0, gamma = 0.01)
	svm_2 = SVC(kernel = "rbf", C = 4.0, gamma = 0.01)
	svm_3 = SVC(kernel = "rbf", C = 4.0, gamma = 0.01)
	xTrain_1 = np.copy(xTrain[:, column_mouth])
	yTrain_1 = np.copy(yTrain)
	#for i in range(len(yTrain_1)):
	#	if yTrain_1[i] not in level_1:
	#		yTrain_1[i] = fake_1
	svm_1.fit(xTrain_1, yTrain_1)
	xTest_1 = np.copy(xTest[:, column_mouth])
	yPredict = svm_1.predict(xTest_1)
	if yPredict[0] in level_1:
		return yPredict
		
	mark2 = []
	for i in range(len(yTrain)):
		if yTrain[i] in level_2 or yTrain[i] in level_3:
			mark2.append(i)
	xTrain_2 = np.copy(xTrain[:, column_mouth + column_eyes])
	xTrain_2 = xTrain_2[mark2,:]
	yTrain_2 = np.copy(yTrain[mark2])
	#for i in range(len(yTrain_2)):
	#	if yTrain_2[i] not in level_2:
	#		yTrain_2[i] = fake_2
	xTest_2 = np.copy(xTest[:, column_mouth + column_eyes])
	svm_2.fit(xTrain_2, yTrain_2)
	yPredict = svm_2.predict(xTest_2)
	if yPredict[0] in level_2:
		return yPredict
	
	mark3 = []
	for i in range(len(yTrain)):
		if yTrain[i] in level_3:
			mark3.append(i)
	xTrain_3 = np.copy(xTrain[:, column_mouth + column_eyes + column_nose])
	xTrain_3 = xTrain_3[mark3,:]
	yTrain_3 = np.copy(yTrain[mark3])
	xTest_3 = np.copy(xTest[:, column_mouth + column_eyes + column_nose])
	svm_3.fit(xTrain_3, yTrain_3)
	yPredict = svm_3.predict(xTest_3)
	return yPredict

def LOO_threeLevelSvm(data):
	loo = cross_validation.LeaveOneOut(len(data))
	totalAccurateNum = 0
	for trainIndex, testIndex in loo:
		xTrain, yTrain = data[trainIndex, 0:43], data[trainIndex, 43]
		xTest, yTest = data[testIndex, 0:43], data[testIndex, 43]
		yPredict = threeLevelSvm(xTrain, yTrain, xTest, yTest)
		if yTest == yPredict :
			totalAccurateNum += 1
		else :
			print yTest, yPredict
	print float(totalAccurateNum) / len(data)
	return

def columnSvm(data, columns):
	loo = cross_validation.LeaveOneOut(len(data))
	totalAccurateNum = 0
	singleExpression = [0] * 7
	singleExpressionAccurate = [0] * 7
	eachAccuracy = [0.0] * 7
	for i in range(1, 8):
		singleExpression[i - 1] = sum(data[:, 43] == i)
	
	for trainIndex, testIndex in loo:
		xTrain, yTrain = data[trainIndex, 0 : 43], data[trainIndex, 43]
		xTest, yTest = data[testIndex, 0 : 43], data[testIndex, 43]
		xTrain = xTrain[:, columns]
		xTest = xTest[:, columns]
		svm = SVC(kernel = "rbf", C = 4.0, gamma = 0.01)
		svm.fit(xTrain, yTrain)
		yPredict = svm.predict(xTest)
		
		if yTest == yPredict :
			totalAccurateNum += 1
			singleExpressionAccurate[yTest[0] - 1] += 1
		else :
			print yTest, yPredict
	print "total accuracy: ", float(totalAccurateNum) / len(data)
	for i in range(0, 7):
		eachAccuracy[i] = float(singleExpressionAccurate[i]) / singleExpression[i]
		print i+1, eachAccuracy[i]
	
	return totalAccurateNum, eachAccuracy

def testColumnSvm(data):
	column_mouth = range(9, 34);
	column_eyes = [0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42]
	column_nose = [8, 35, 36, 37, 38, 39]
	columnSvm(data, column_mouth)
	columnSvm(data, column_mouth + column_eyes)
	columnSvm(data, column_mouth + column_eyes + column_nose)
	return

def twoLevelSvm(xTrain, yTrain, xTest, yTest, level_1):
	#level_1 = [2, 4, 1]
	level_2 = []
	for i in range(1, 8):
		if i not in level_1 :
			level_2.append(i)
	fake_1 = 4
	column_mouth = range(9, 34);
	column_eyes = [0, 1, 2, 3, 4, 5, 6, 7, 40, 41, 42]
	column_nose = [8, 35, 36, 37, 38, 39]
	svm_1 = SVC(kernel = "rbf", C = 4.0, gamma = 0.01)
	svm_2 = SVC(kernel = "rbf", C = 4.0, gamma = 0.01)
	xTrain_1 = np.copy(xTrain[:, column_mouth])
	yTrain_1 = np.copy(yTrain)
	#for i in range(len(yTrain_1)):
	#	if yTrain_1[i] not in level_1:
	#		yTrain_1[i] = fake_1
	svm_1.fit(xTrain_1, yTrain_1)
	xTest_1 = np.copy(xTest[:, column_mouth])
	yPredict = svm_1.predict(xTest_1)
	if yPredict[0] in level_1:
		return yPredict
		
	mark2 = []
	for i in range(len(yTrain)):
		if yTrain[i] in level_2:
			mark2.append(i)
	xTrain_2 = np.copy(xTrain[:, column_mouth + column_eyes + column_nose])
	xTrain_2 = xTrain_2[mark2,:]
	yTrain_2 = np.copy(yTrain[mark2])
	#for i in range(len(yTrain_2)):
	#	if yTrain_2[i] not in level_2:
	#		yTrain_2[i] = fake_2
	xTest_2 = np.copy(xTest[:, column_mouth + column_eyes + column_nose])
	svm_2.fit(xTrain_2, yTrain_2)
	yPredict = svm_2.predict(xTest_2)
	return yPredict

def LOO_twoLevelSvm(data, level_1):
	loo = cross_validation.LeaveOneOut(len(data))
	totalAccurateNum = 0
	for trainIndex, testIndex in loo:
		xTrain, yTrain = data[trainIndex, 0:43], data[trainIndex, 43]
		xTest, yTest = data[testIndex, 0:43], data[testIndex, 43]
		yPredict = twoLevelSvm(xTrain, yTrain, xTest, yTest, level_1)
		if yTest == yPredict :
			totalAccurateNum += 1
		#else :
			#print yTest, yPredict
	accuracy = float(totalAccurateNum) / len(data)
	print accuracy
	return accuracy
	
def tuneTwoLevelSvm(data):
	level_1 = []
	'''
	for i in range(1, 5):
		level_1.append(i)
		for j in range(i + 1, 6):
			level_1.append(j)
			for k in range(j + 1, 7):
				level_1.append(k)
				for l in range(k + 1, 8):
					level_1.append(l)
					print level_1
					LOO_twoLevelSvm(data, level_1)
					level_1.pop()
				level_1.pop()
			level_1.pop()
		level_1.pop()
	'''
	'''
	for i in range(1, 6):
		level_1.append(i)
		for j in range(i + 1, 7):
			level_1.append(j)
			for k in range(j + 1, 8):
				level_1.append(k)
				print level_1
				LOO_twoLevelSvm(data, level_1)
				level_1.pop()
			level_1.pop()
		level_1.pop()
	'''
	'''
	for j in range(1, 7):
		level_1.append(j)
		for k in range(j + 1, 8):
			level_1.append(k)
			print level_1
			LOO_twoLevelSvm(data, level_1)
			level_1.pop()
		level_1.pop()
	'''	
	for k in range(1, 8):
		level_1.append(k)
		print level_1
		LOO_twoLevelSvm(data, level_1)
		level_1.pop()
	return
	
def main(argv):
	data=loadDataSet('CKPlus_AUEmotion_DataSet.csv')
	#print data
	#simpleAnalyze(data)
	#tuneSvm(data)
	LOO_threeLevelSvm(data)
	#testColumnSvm(data)
	#LOO_twoLevelSvm(data)
	#tuneTwoLevelSvm(data)
	
	
if __name__ == "__main__":
	main(sys.argv[1:])