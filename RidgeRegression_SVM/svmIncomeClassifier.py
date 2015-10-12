#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__='Chengjun Yuan @UVa'

import os
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

def loadDataSet(fileName):  
	absPath=os.path.abspath(fileName)
	data=np.loadtxt(absPath,delimiter=', ',dtype ='S')
	return data

def normalizeData(class_):
	max=np.amax(class_)
	min=np.amin(class_)
	newClass=[]
	for i in range(len(class_)):
		new_=(float(class_[i])-min)*1.0/float(max-min)
		newClass.append(new_)
	return np.asarray(newClass)
	
def preprocessClass(le,class_):
	List=set(class_)
	class_items=list(List)
	le.fit(class_items)
	class_=le.transform(class_)
	if '?' in List:
		class_=class_[:,None]
		#find the corresponding number of missed item
		missed=le.transform('?')
		#imputer the missed value with the average/mean value 
		imputer=Imputer(missing_values=missed,strategy="mean")
		class_=imputer.fit_transform(class_)[:,0]
		#shift float to int
		class_=np.rint(class_)
		#LabelEncoder again to fix the removed missed item
		le.fit(np.unique(class_))
		class_=le.transform(class_)
	class_=normalizeData(class_)
	return class_

def preprocessDataSet(fileName):
	data=loadDataSet(fileName)
	le = preprocessing.LabelEncoder()
	#prepocess class age
	class_age=data[:,0]
	class_age=map(int,class_age)
	class_age=normalizeData(class_age)
	#prepocess class work class
	class_work=data[:,1]
	class_work=preprocessClass(le,class_work)
	#prepocess class fnlwgt
	class_fnlwgt=data[:,2]
	class_fnlwgt=map(int,class_fnlwgt)
	class_fnlwgt=normalizeData(class_fnlwgt)
	#prepocess class education
	class_edu=data[:,3]
	class_edu=preprocessClass(le,class_edu)
	#prepocess class eduNum
	class_eduNum=data[:,4]
	class_eduNum=map(int,class_eduNum)
	class_eduNum=normalizeData(class_eduNum)
	#preprocess class mariSta
	class_mari=data[:,5]
	class_mari=preprocessClass(le,class_mari)
	#preprocess class occupation
	class_occu=data[:,6]
	class_occu=preprocessClass(le,class_occu)
	#preprocess class relationship
	class_rela=data[:,7]
	class_rela=preprocessClass(le,class_rela)
	#preprocess class race
	class_race=data[:,8]
	class_race=preprocessClass(le,class_race)
	#preprocess class sex
	class_sex=data[:,9]
	class_sex=preprocessClass(le,class_sex)
	#prepocess class capital-gain
	class_capg=data[:,10]
	class_capg=map(int,class_capg)
	class_capg=normalizeData(class_capg)
	#prepocess class capital-loss
	class_capl=data[:,11]
	class_capl=map(int,class_capl)
	class_capl=normalizeData(class_capl)
	#prepocess class hours-per-week
	class_hpw=data[:,12]
	class_hpw=map(int,class_hpw)
	class_hpw=normalizeData(class_hpw)
	#preprocess class native-country
	class_nati=data[:,13]
	class_nati=preprocessClass(le,class_nati)
	#preprocess class income
	class_income=data[:,14]
	List=set(class_income)
	class_items=list(List)
	le.fit(class_items)
	class_income=le.transform(class_income)
	#combine the classes to output
	trainSet=np.concatenate((class_age[:,None],class_work[:,None],class_fnlwgt[:,None],class_edu[:,None],class_eduNum[:,None],class_mari[:,None],class_occu[:,None],class_rela[:,None],class_race[:,None],class_sex[:,None],class_capg[:,None],class_capl[:,None],class_hpw[:,None],class_nati[:,None]),axis=1)
	return trainSet,class_income
	
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

def SearchParas(train,test):
	print "reading train set"
	trainSet,trainResult=preprocessDataSet(train)
	print "reading test set"
	testSet,testResult=preprocessDataSet(test)
	for C_ in [1,10,20.0]:
		for gamma_ in [0.0,0.1,0.2,0.5,1.0]:
			print "C= ",C_," gamma= ",gamma_
			svmPredict=SVC(C=C_,gamma=gamma_,kernel='rbf')
			svmPredict.fit(trainSet,trainResult)
			predictResult=svmPredict.predict(trainSet)
			trainAccuracy=getAccuracy(predictResult,trainResult)
			print "trainAccuracy",trainAccuracy
			predictResult=svmPredict.predict(testSet)
			testAccuracy=getAccuracy(predictResult,testResult)
			print "testAccuracy",testAccuracy
	return 0
	
#print loadDataSet("data\\adult.data")
#processDataSet("adult.data","adult.test")[0:10]
#SearchParas("adult.data","adult.test")