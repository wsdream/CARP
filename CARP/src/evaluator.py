########################################################
# evaluator.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/11/03
########################################################

import numpy as np 
from numpy import linalg as LA
import time, sys
import random
import core
from utilities import *


########################################################
# Function to run the prediction approach at each density
# 
def execute(tensor, density, para):
    startTime = time.clock()
    startTime = time.clock()
    [numUser, numService, numTime] = tensor.shape
    rounds = para['rounds']
    logger.info('Data size: %d users * %d services * %d timeslices'\
    	%(numUser, numService, numTime))
    logger.info('Run the algorithm for %d rounds: density = %.2f.'%(rounds, density))
    evalResults = np.zeros((rounds, len(para['metrics']))) 
    timeResults = np.zeros((rounds, 1))
    
    for k in range(rounds):
		logger.info('----------------------------------------------')
		logger.info('%d-round starts.'%(k + 1))
		logger.info('----------------------------------------------')

		# remove the entries of data to generate trainTensor and testTensor
		(trainTensor, testTensor) = removeTensor(tensor, density, k, para)
		logger.info('Removing data entries done.')

		# invocation to the prediction function
		iterStartTime = time.clock() # to record the running time for one round             
		predictedTensor = core.predict(trainTensor, para) 
		timeResults[k] = time.clock() - iterStartTime

		# calculate the prediction error
		result = np.zeros((numTime, len(para['metrics'])))
		for i in range(numTime):
			testMatrix = testTensor[:, :, i]
			predictedMatrix = predictedTensor[:, :, i]
			(testVecX, testVecY) = np.where(testMatrix)
			testVec = testMatrix[testVecX, testVecY]
			predVec = predictedMatrix[testVecX, testVecY]
			result[i, :] = errMetric(testVec, predVec, para['metrics'])		
		evalResults[k, :] = np.average(result, axis=0)

		logger.info('%d-round done. Running time: %.2f sec'%(k + 1, timeResults[k]))
		logger.info('----------------------------------------------')

    outFile = '%savg_%sResult_%.2f.txt'%(para['outPath'], para['dataType'], density)
    saveResult(outFile, evalResults, timeResults, para)

    logger.info('Density = %.2f done. Running time: %.2f sec'
			%(density, time.clock() - startTime))
    logger.info('==============================================')
########################################################


########################################################
# Function to remove the entries of data tensor
# Return the trainTensor and the corresponding testTensor
#
def removeTensor(tensor, density, round, para):
	numTime = tensor.shape[2]
	trainTensor = np.zeros(tensor.shape)
	testTensor = np.zeros(tensor.shape)
	for i in range(numTime):
		seedID = round + i * 100
		(trainMatrix, testMatrix) = removeEntries(tensor[:, :, i], density, seedID)
		trainTensor[:, :, i] = trainMatrix
		testTensor[:, :, i] = testMatrix
	return trainTensor, testTensor
########################################################


########################################################
# Function to remove the entries of data matrix
# Use guassian random sampling
# Return trainMatrix and testMatrix
#
def removeEntries(matrix, density, seedID):
	numAll = matrix.size
	numTrain = int(numAll * density)
	(vecX, vecY) = np.where(matrix > -1000)
	np.random.seed(seedID % 100)
	randPermut = np.random.permutation(numAll)	
	np.random.seed(seedID)
	randSequence = np.random.normal(0, numAll / 6.0, numAll * 50)

	trainSet = []
	flags = np.zeros(numAll)
	for i in xrange(randSequence.shape[0]):
		sample = int(abs(randSequence[i]))
		if sample < numAll:
			idx = randPermut[sample]
			if flags[idx] == 0 and matrix[vecX[idx], vecY[idx]] > 0:
				trainSet.append(idx)
				flags[idx] = 1
		if len(trainSet) == numTrain:
			break
	if len(trainSet) < numTrain:
		logger.critical('Exit unexpectedly: not enough data for density = %.2f.', density)
		sys.exit()

	trainMatrix = np.zeros(matrix.shape)
	trainMatrix[vecX[trainSet], vecY[trainSet]] = matrix[vecX[trainSet], vecY[trainSet]]
	testMatrix = np.zeros(matrix.shape)
	testMatrix[matrix > 0] = matrix[matrix > 0]
	testMatrix[vecX[trainSet], vecY[trainSet]] = 0

    # ignore invalid testing users or services             
	idxX = (np.sum(trainMatrix, axis=1) == 0)
	testMatrix[idxX, :] = 0
	idxY = (np.sum(trainMatrix, axis=0) == 0)
	testMatrix[:, idxY] = 0    
	return trainMatrix, testMatrix
########################################################


########################################################
# Function to compute the evaluation metrics
#
def errMetric(realVec, predVec, metrics):
    result = []
    absError = np.abs(predVec - realVec) 
    mae = np.sum(absError)/absError.shape
    for metric in metrics:
	    if 'MAE' == metric:
			result = np.append(result, mae)
	    if 'NMAE' == metric:
		    nmae = mae / (np.sum(realVec) / absError.shape)
		    result = np.append(result, nmae)
	    if 'RMSE' == metric:
		    rmse = LA.norm(absError) / np.sqrt(absError.shape)
		    result = np.append(result, rmse)
	    if 'MRE' == metric or 'NPRE' == metric:
	        relativeError = absError / realVec
	        relativeError = np.sort(relativeError)
	        if 'MRE' == metric:
		    	mre = np.median(relativeError)
		    	result = np.append(result, mre)
	        if 'NPRE' == metric:
		    	npre = relativeError[np.floor(0.9 * relativeError.shape[0])] 
		    	result = np.append(result, npre)
    return result
########################################################

