########################################################
# core.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/10/22
########################################################

import numpy as np
import sys
from utilities import *


#########################################################
# Function to perform the prediction algorithm
#
def predict(removedTensor, para):
	# tMean: mean value along the time
    numTimeSlice = removedTensor.shape[2]
    tMeanMatrix = np.sum(removedTensor, axis=2) /\
    	(np.sum(removedTensor > 0, axis=2) + np.spacing(1))
    predTensor = np.rollaxis(np.tile(tMeanMatrix, (numTimeSlice, 1, 1)), 0, 3)
    
    # mean value of the slice
    for i in xrange(numTimeSlice):
    	removedMatrix = removedTensor[:, :, i]
    	sliceMean = np.sum(removedMatrix) / (np.sum(removedMatrix > 0) + np.spacing(1))
    	predTensor[:, :, i] = np.where(predTensor[:, :, i] > 0, predTensor[:, :, i], sliceMean)

    return predTensor
#########################################################  

