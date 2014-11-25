########################################################
# core.pyx
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/10/30
########################################################

import time
import numpy as np
cimport numpy as np # import C-API
from libcpp cimport bool
import scipy.cluster.vq # import k-means
from utilities import *


#########################################################
# Make declarations on functions from cpp file
#
cdef extern from "PMF.h":
    void PMF(double *removedData, int numUser, int numService, 
        int dim, double lmda, int maxIter, double etaInit, 
        double *Udata, double *Sdata)
#########################################################


#########################################################
# Function to perform the prediction algorithm
# Wrap up the C++ implementation
#
def predict(removedTensor, para):
    (numUser, numService, numTimeSlice) = removedTensor.shape
    numContext = para['numContext']

    # context clustering
    featureMatrix = np.sum(removedTensor, axis=0) / (np.sum(removedTensor > 0, axis=0) + np.spacing(1))
    clusterCxt = [[] for i in xrange(numContext)]
    [_, attrCxt] = scipy.cluster.vq.kmeans2(featureMatrix.T, numContext, minit = 'points')
    for i in xrange(numTimeSlice):
        clusterCxt[attrCxt[i]].append(i)
    logger.info('Context clustering done.')
    
    # data aggregation for each context    
    cxtTensor = np.zeros((numUser, numService, numContext))
    countTensor = np.zeros((numUser, numService, numContext))
    for i in xrange(numContext):
        cxtTensor[:, :, i] = np.sum(removedTensor[:, :, clusterCxt[i]], axis=2)
        countTensor[:, :, i] = np.sum(removedTensor[:, :, clusterCxt[i]] > 0, axis=2)
    cxtTensor = cxtTensor / (countTensor + np.spacing(1))

    # initialization
    cdef int dim = para['dimension']
    cdef double lmda = para['lambda']
    cdef int maxIter = para['maxIter']
    cdef double etaInit = para['etaInit']
    cdef np.ndarray[double, ndim=2, mode='c'] U = np.random.rand(numUser, dim)        
    cdef np.ndarray[double, ndim=2, mode='c'] S = np.random.rand(numService, dim)     
    predCxtTensor = np.zeros((numUser, numService, numContext))

    # context-aware matrix factorization
    for i in xrange(numContext):
        removedMatrix = cxtTensor[:, :, i].copy()
	    # wrap up PMF.cpp
        PMF(
            <double *> (<np.ndarray[double, ndim=2, mode='c']> removedMatrix).data,
            numUser,
            numService,
            dim,
            lmda,
            maxIter,
            etaInit,
            <double *> U.data,
            <double *> S.data
            )
        predCxtTensor[:, :, i] = np.dot(U, S.T)
    predCxtTensor[cxtTensor > 0] = cxtTensor[cxtTensor > 0]

    # context-specific prediction
    predTensor = np.zeros((numUser, numService, numTimeSlice))
    for i in xrange(numContext):
        predTensor[:, :, clusterCxt[i]] = np.rollaxis(np.tile(predCxtTensor[:, :, i], 
            (len(clusterCxt[i]), 1, 1)), 0, 3)
    predTensor[removedTensor > 0] = removedTensor[removedTensor > 0]
    
    return predTensor
#########################################################  
