'''
emotions:
	0=neutral, 
	1=anger, 
	2=contempt, 
	3=disgust, 
	4=fear, 
	5=happy, 
	6=sadness, 
	7=surprise
'''
import subprocess
import shlex
import os,fnmatch
import numpy as np
import csv

def findFiles(pattern,path):
	result=[]
	for root, dirs, files in os.walk(path):
		for name in files:
			if fnmatch.fnmatch(name, pattern):
				result.append(os.path.join(root, name))
	return result

def buildDataSet(emotionFilesPaths):
	#dataSet=numpy.zeros((327,44))
	dataSet=[]
	linesNames=[]
	for i in range(len(emotionFilesPaths)):
		facsFilePath=emotionFilesPaths[i].replace('Emotion','FACS')
		facsFilePath=facsFilePath.replace('emotion','facs')
		name=facsFilePath[facsFilePath.rindex('\\')+1:facsFilePath.rindex('_')]
		#print emotionFilesPaths[i], facsFilePath, name
		emotion=np.genfromtxt(emotionFilesPaths[i])
		facs=np.genfromtxt(facsFilePath)
		linesNames.append(name)
		dataLine=[0]*45
		dataLine[0]=name
		dataLine[44]=emotion[()]
		print facsFilePath, facs.ndim
		if facs.ndim==2:
			for j in range(len(facs)):
				AU_index=int(facs[j][0])
				if AU_index<44:
					dataLine[AU_index]=int(facs[j][1])
				else:
					print facs[j]
		else:
			AU_index=int(facs[0])
			if AU_index<44:
				dataLine[AU_index]=int(facs[1])
			else:
				print facs
		dataSet.append(dataLine)
		#if i>5:
		#	break
	return dataSet

def writeDataToCSV(dataSet):
	writer=csv.writer(open("CKPlus_AUEmotion_DataSet.csv", 'wb'),delimiter=',')
	writer.writerows(dataSet)
		
print 'create a single dataset from Cohn Kanade plus dataset...'
emotionFilesPaths=findFiles('*.txt', 'Emotion')
print len(emotionFilesPaths)
dataSet=buildDataSet(emotionFilesPaths)
writeDataToCSV(dataSet)

