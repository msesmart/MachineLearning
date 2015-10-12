# linear Regression HW1 MachineLearning
# -*- coding: utf-8 -*-
__author__='Chengjun Yuan @UVa'

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
	data=np.genfromtxt(fileName)
	xVal=data[:,1]
	yVal=data[:,2]
	fig=plt.plot(xVal,yVal,'ro')
	plt.xlabel('xVal')
	plt.ylabel('yVal')
	plt.title(fileName)
	plt.show()
	return xVal,yVal
	
def standRegres(xVal,yVal):
	temp=[1]*len(xVal)
	xm=np.vstack((temp,xVal))
	xm=np.matrix(xm.transpose())
	xm=(xm.transpose()*xm).getI()*xm.transpose()
	ym=np.matrix(yVal).transpose()
	theta=np.squeeze(np.asarray((xm*ym).transpose()))
	# plot original data & linear regression line
	t=np.arange(min(xVal),max(xVal),(max(xVal)-min(xVal))/4.03)
	plt.plot(xVal,yVal,'ro',t,theta[0]+theta[1]*t,'b-',linewidth=2.0)
	plt.xlabel('xVal')
	plt.ylabel('yVal')
	plt.show()
	return theta
