'''
	Author:	Chengjun Yuan   <cy3yb@virginia.edu>
	Time:	Dec.06 2015
	This code is to implement K means clustering and 
	Gaussian Mixture clustering method.	
'''

import os
import sys
import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def loadData(fileName):
	print "load data"
	data = np.loadtxt(fileName)
	xData = data[:, range(0, len(data[0]) - 1)]
	yData = data[:, len(data[0]) - 1]
	return xData, yData

def objectFunction(clusters, label, X):
	object = 0.0
	for i in range(len(X)):
		index = label[i]
		object += math.pow(LA.norm(X[i] - clusters[index]), 2.0)
	return object

# Q5
def purity(label, Y):
	totalNum = 0
	uniqueLabel = np.unique(label)
	for k in range(len(uniqueLabel)):
		index = np.where(label == k)[0]
		localY = Y[index]
		uniqueLocalY, counts = np.unique(localY, return_counts = True)
		maxLocalY = np.amax(counts)
		totalNum += maxLocalY
	return float(totalNum) / len(Y)
	
def kmeans(X, k, maxIter):
	clusters = X[np.random.choice(range(len(X)), k, replace=False)]
	iteration = 0
	while iteration < maxIter:
		preClusters = clusters.copy()
		label = []
		for x in X:
			label.append(np.argmin([LA.norm(x - c) for c in preClusters]))
		for i in range(k):
			clusters[i] = sum(X[np.array(label) == i]) / len(X[np.array(label) == i])
		if np.all(clusters == preClusters):
			break
		iteration = iteration + 1
	return np.array(label), clusters
	
def e_step(X, mu, p, sigma, K): 
	gamma = np.zeros((len(X), K))
	for i in range(len(X)):
		x = X[i]
		accum = np.zeros(K)
		for h in range(K):
			accum[h] = np.exp(-0.5 * np.inner(np.inner(x-mu[h], np.linalg.inv(sigma)), x-mu[h])) * p[h]
			gamma[i] = accum / sum(accum)
	return gamma

def m_step(X, gamma, K):  
	mu = np.zeros((K, len(X[0])))
	p = np.zeros(K)
	for h in range(K):
		numerator = np.zeros(len(X[0]))
		denominator = 0.0
		for i in range(len(X)):
			numerator += gamma[i, h] * X[i]
			denominator += gamma[i, h]
		mu[h] = numerator / denominator
		p[h] = denominator / len(X)
	return mu, p

def gmmCluster(X, K, covType, maxIter):
	if covType == 'diag':
		sigma = np.diag(np.diag(np.cov(X.T)))
	elif covType == 'full':
		sigma = np.cov(X.T)
	else :
		print "Error: Invalid covType, either diag or full."
		sys.exit()
		
	# initiate mu, p
	mu = X[np.random.choice(range(len(X)), K, replace=False)]
	p = np.ones(K) / K
	iteration = 0
	while iteration < maxIter:
		preMu = mu.copy()
		preP = p.copy()
		gamma = e_step(X, mu, p, sigma, K)
		mu, p = m_step(X, gamma, K)
		if np.sum(abs(mu - preMu)) / np.sum(abs(preMu)) < 0.0001:
			break
		iteration += 1
	label = [np.argmax(g) for g in gamma]
	return np.array(label), mu



if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage: python clustering.py dataSetDirectoryFullPath"
		sys.exit()
		
	dataSetDirectoryFullPath = sys.argv[1]
	fileName = os.path.join(dataSetDirectoryFullPath, "humanData.txt")
	# Q3 
	X, Y = loadData(fileName)
	'''
	label, clusters = kmeans(X, 2, 1000)
	plt.scatter(X[:, 0], X[:, 1], c = label, alpha = 1.0)
	plt.show()
	puri = purity(label, Y)
	print 'purityMetric for Q3 is ', puri
	'''
	'''
	# Q4
	objects = []
	for k in range(1, 7):
		label, clusters = kmeans(X, k, 1000)
		objects.append(objectFunction(clusters, label, X))
	fig, axes = plt.plot(range(1, 7), objects, '-o', linewidth = 2)
	axes.set_xlabel("K")
	axes.set_ylabel("Object Function")
	plt.show()
	'''
	'''
	# Q7
	label, mu = gmmCluster(X, 2, 'diag', 1000)
	plt.scatter(X[:, 0], X[:, 1], c = label, alpha = 1.0)
	plt.show()
	puri = purity(label, Y)
	print 'purityMetric for Q7 covType=diag is ', puri
	
	label, mu = gmmCluster(X, 2, 'full', 1000)
	plt.scatter(X[:, 0], X[:, 1], c = label, alpha = 1.0)
	plt.show()
	puri = purity(label, Y)
	print 'Q9 purityMetric for Q7 covType=full is ', puri
	'''
	# Q8
	fileName = os.path.join(dataSetDirectoryFullPath, "audioData.txt")
	X, Y = loadData(fileName)
	print len(X[0])
	label, mu = gmmCluster(X, 2, 'diag', 1000)
	plt.scatter(X[:, 0], X[:, 1], c = label, alpha = 1.0, edgecolors = 'face')
	plt.show()
	puri = purity(label, Y)
	print 'Q9 purityMetric for Q8 covType=diag is ', puri

