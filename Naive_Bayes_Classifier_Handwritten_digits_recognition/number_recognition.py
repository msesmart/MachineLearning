#!/usr/bin/env python

import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA

def loadData(file):
	print "load data"
	data = np.loadtxt(file)
	xData = data[:, 1:256]
	yData = data[:, 0]
	return xData, yData

def decision_tree(train, test):
	y = []
	xTrain, yTrain = loadData(train)
	xTest, yTest = loadData(test)
	dt = DecisionTreeClassifier(criterion = "entropy", splitter = "best", max_features = "sqrt")
	dt.fit(xTrain, yTrain)
	y = dt.predict(xTest)
	testError = 1 - dt.score(xTest, yTest)
	print 'Test error: ' , testError
	return y

def knn(train, test):
	y = []
	xTrain, yTrain = loadData(train)
	xTest, yTest = loadData(test)
	kNN = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
	kNN.fit(xTrain, yTrain)
	y = kNN.predict(xTest)
	testError = 1 - kNN.score(xTest, yTest)
	trainError =  1 - kNN.score(xTrain, yTrain)
	print 'Test error: ' , testError
	print 'Training error: ' , trainError
	return y

def neural_net(train, test):
	y = []
	xTrain, yTrain = loadData(train)
	xTest, yTest = loadData(test)
	nN = Perceptron()
	nN.fit(xTrain, yTrain)
	y = nN.predict(xTest)
	testError = 1 - nN.score(xTest, yTest)
	print 'Test error: ' , testError
	return y

def svm(train, test):
	y = []
	xTrain, yTrain = loadData(train)
	xTest, yTest = loadData(test)
	for i in ['rbf', 'poly', 'linear'] :
		print "svm kernel: ", i
		svm = SVC(kernel = i)
		svm.fit(xTrain, yTrain)
		y = svm.predict(xTest)
		testError = 1 - svm.score(xTest, yTest)
		print 'Test error: ' , testError
	return y

def pca_knn(train, test):
	y = []
	xTrain, yTrain = loadData(train)
	xTest, yTest = loadData(test)
	for i in [32, 64, 128] :
		print "n_components", i
		pca = RandomizedPCA(n_components = i, random_state = 1)
		pca.fit(xTrain)
		reducedXTrain = pca.transform(xTrain)
		reducedXTest = pca.transform(xTest)
		kNN = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
		kNN.fit(reducedXTrain, yTrain)
		y = kNN.predict(reducedXTest)
		testError = 1 - kNN.score(reducedXTest, yTest)
		print 'Test error: ' , testError
		print "sum of explained_variance_ratio_", pca.explained_variance_ratio_.sum()
	return y

def pca_svm(train, test):
	y = []
	xTrain, yTrain = loadData(train)
	xTest, yTest = loadData(test)
	for i in [32, 64, 128] :
		print "n_components", i
		pca = RandomizedPCA(n_components = i, random_state = 1)
		pca.fit(xTrain)
		reducedXTrain = pca.transform(xTrain)
		reducedXTest = pca.transform(xTest)
		
		svm = SVC(kernel = 'poly')
		svm.fit(reducedXTrain, yTrain)
		y = svm.predict(reducedXTest)
		testError = 1 - svm.score(reducedXTest, yTest)
		print 'Test error: ' , testError
		print "sum of explained_variance_ratio_", pca.explained_variance_ratio_.sum()
	return y

if __name__ == '__main__':
	model = sys.argv[1]
	train = sys.argv[2]
	test = sys.argv[3]

	if model == "dtree":
		print(decision_tree(train, test))
	elif model == "knn":
		print(knn(train, test))
	elif model == "net":
		print(neural_net(train, test))
	elif model == "svm":
		print(svm(train, test))
	elif model == "pcaknn":
		print(pca_knn(train, test))
	elif model == "pcasvm":
		print(pca_svm(train, test))
	else:
		print("Invalid method selected!")
