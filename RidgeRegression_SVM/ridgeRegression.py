# linear Regression HW1 MachineLearning
# -*- coding: utf-8 -*-
__author__='Chengjun Yuan @UVa'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random

def loadDataSet(fileName):
	data=np.genfromtxt(fileName)
	xVal=data[:,0:3]
	yVal=data[:,3]
	
	"""fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xVal[:,1],xVal[:,2],yVal,c='r',marker='o')
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('y')
	plt.show() """
	return xVal,yVal
	
def ridgeRegressx1x2(xVal,lambda_):
	yVal=xVal[:,2]
	xVal=xVal[:,0:2]
	xm=np.matrix(xVal)
	xtx=xm.transpose()*xm
	lambda_I=np.identity(2)*lambda_
	#print lambda_I
	xm=(xtx+lambda_I).getI()*xm.transpose()
	ym=np.matrix(yVal).transpose()
	theta=np.squeeze(np.asarray((xm*ym).transpose()))
	
	print theta
	t=np.arange(min(xVal[:,1]),max(xVal[:,1]),(max(xVal[:,1])-min(xVal[:,1]))/4.03)
	plt.plot(xVal[:,1],yVal,'ro',t,theta[0]+theta[1]*t,'b-',linewidth=2.0)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.show()

	return theta	

def ridgeRegress(xVal,yVal,lambda_):
	xm=np.matrix(xVal)
	xtx=xm.transpose()*xm
	lambda_I=np.identity(3)*lambda_
	#print lambda_I
	xm=(xtx+lambda_I).getI()*xm.transpose()
	ym=np.matrix(yVal).transpose()
	theta=np.squeeze(np.asarray((xm*ym).transpose()))
	'''
	print theta
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xVal[:,1],xVal[:,2],yVal,c='r',marker='o')
	x1=np.arange(min(xVal[:,1]),max(xVal[:,1]),(max(xVal[:,1])-min(xVal[:,1]))/4.03)
	x2=np.arange(min(xVal[:,2]),max(xVal[:,2]),(max(xVal[:,2])-min(xVal[:,2]))/4.03)
	x1,x2=np.meshgrid(x1,x2)
	y=theta[0]+theta[1]*x1+theta[2]*x2
	#ax.plot_surface(x1,x2,y,rstride=4, cstride=4, color='g')
	ax.plot_wireframe(x1,x2,y, rstride=1, cstride=1)
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('y')
	plt.show()
	'''
	return theta

def test(xVal,yVal,theta):
	err=(np.dot(np.transpose(yVal-np.dot(xVal,theta)),yVal-np.dot(xVal,theta)))
	#print err,len(xVal), err/len(xVal)
	return err/len(xVal)

def getTrainTestData(xVal,yVal,List,index):
	foldVolume=int(len(xVal)/10)
	tempTestIndex=range(foldVolume*index,foldVolume*(index+1))
	testIndex=List[tempTestIndex]
	tempAllIndex=range(len(xVal))
	tempTrainIndex=np.delete(tempAllIndex,tempTestIndex)
	#print tempTestIndex, tempTrainIndex
	trainIndex=List[tempTrainIndex]
	return xVal[trainIndex],yVal[trainIndex],xVal[testIndex],yVal[testIndex]
	
def cv(xVal,yVal):
	random.seed(37)
	RSE=[]
	numLambda=50
	Jbeta=0.0
	List=list(range(len(xVal)))
	random.shuffle(List)
	List=np.asarray(List)
	for i in range(1,numLambda+1):
		#print i
		Jbeta=0.0
		for j in range(10):
			xTrain,yTrain,xTest,yTest=getTrainTestData(xVal,yVal,List,j)
			theta=ridgeRegress(xTrain,yTrain,0.02*i)
			Jbeta=Jbeta+test(xTest,yTest,theta)
		#print '$$$$',i*0.02, Jbeta, Jbeta/10.0
		Jbeta=Jbeta/10.0
		RSE.append(Jbeta);
	RSE=np.asarray(RSE)
	lambda_=np.linspace(0.02,numLambda*0.02,numLambda)
	'''
	fig=plt.plot(lambda_,RSE,'r-')
	plt.xlabel('lambda')
	plt.ylabel('10-fold Jbeta')
	plt.show()
	'''
	index=np.where(RSE==RSE.min())
	return float(index[0][0]+1)*0.02


#xVal,yVal=loadDataSet("data\RRdata.txt")
#print cv(xVal,yVal)
#ridgeRegress(xVal,yVal,0.26)
#ridgeRegressx1x2(xVal,0.0)