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
cdef extern from "CARP.h":
    void getSimMatrix(double *removedData, double *simMatrixData, int numTimeSlice, 
        int numInvocations)
    void CARP(double *removedData, double *predData, int numUser, int numService, 
        int numContext, int dim, double lmda, int maxIter, bool debugMode, 
        double *Udata, double *Sdata, double *Cdata)
#########################################################


#########################################################
# Function to perform the prediction algorithm
# Wrap up the C++ implementation
#
def predict(removedTensor, para):
    (numUser, numService, numTimeSlice) = removedTensor.shape
    numContext = para['numContext']

    # compute similarity matrix between time slices
    invocMatrix = removedTensor.reshape(numUser * numService, numTimeSlice).T.copy()
    cdef np.ndarray[double, ndim=2, mode='c'] simMatrix = np.zeros((numTimeSlice, numTimeSlice))
    getSimMatrix(
        <double *> (<np.ndarray[double, ndim=2, mode='c']> invocMatrix).data,
        <double *> simMatrix.data,
        <int> numTimeSlice,
        <int> numUser * numService
        )
    
    # context clustering
    clusterCxt = [[] for i in xrange(numContext)]
    [_, attrCxt] = scipy.cluster.vq.kmeans2(simMatrix, numContext, minit = 'points')
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
    cdef bool debugMode = para['debugMode']
    cdef int dim = para['dimension']
    cdef double lmda = para['lambda']
    cdef int maxIter = para['maxIter']
    cdef np.ndarray[double, ndim=3, mode='c'] predCxtTensor =\
        np.zeros((numUser, numService, numContext))
    cdef np.ndarray[double, ndim=2, mode='c'] U = np.random.rand(numUser, dim)        
    cdef np.ndarray[double, ndim=2, mode='c'] S = np.random.rand(numService, dim)
    cdef np.ndarray[double, ndim=3, mode='c'] C = np.random.rand(dim, dim, numContext)

    # context-aware matrix factorization
    # wrap up CARP.cpp
    CARP(<double *> (<np.ndarray[double, ndim=3, mode='c']> cxtTensor).data,
        <double *> predCxtTensor.data,
        <int> numUser,
        <int> numService,
        <int> numContext,
        dim,
        lmda,
        maxIter,
        debugMode,
        <double *> U.data,
        <double *> S.data,
        <double *> C.data
        )

    # context-specific prediction
    predTensor = np.zeros((numUser, numService, numTimeSlice))
    for i in xrange(numContext):
        predTensor[:, :, clusterCxt[i]] = np.rollaxis(np.tile(predCxtTensor[:, :, i], 
            (len(clusterCxt[i]), 1, 1)), 0, 3)
    predTensor[removedTensor > 0] = removedTensor[removedTensor > 0]

    return predTensor
#########################################################  
