/********************************************************
 * CARP.h: header file of CARP.cpp
 * Author: Jamie Zhu <jimzhu@GitHub>
 * Created: 2014/5/6
 * Last updated: 2014/7/15
********************************************************/

#include <algorithm>
#include <iostream>
using namespace std;

typedef pair<double, double> pdd;
const double eps = 1e-10;

/* Compute the similarity matrix between time slices */
void getSimMatrix(double *removedData, double *simMatrixData, int numTimeSlice, 
    int numInvocations);

/* Perform the core approach of CARP */
void CARP(double *removedData, double *predData, int numUser, int numService, 
	int numContext, int dim, double lmda, int maxIter, bool debugMode, 
	double *Udata, double *Sdata, double *Cdata);

/* Compute pcc value between two vectors */
double getPCC(double *uA, double *uB, double meanA, double meanB, int numInvocations);

/* Update the corresponding R_hat */
void updateR_hat(bool flag, double ***R, double ***R_hat, double **U, double **S, 
    double ***C, int numUser, int numService, int numContext, int dim);

/* Compute the loss value of CARP */
pdd lossFunction(double ***R, double ***R_hat, double **U, double **S, double ***C, 
    int numUser, int numService, int numContext, int dim, double lmda);

/* Transform a vector into a matrix */ 
double **vector2Matrix(double *vector, int row, int col);

/* Transform a vector into a 3D tensor */ 
double ***vector2Tensor(double *vector, int row, int col, int height);

/* Allocate memory for a 1D array */
double *createVector(int size);

/* Free memory for a 1D array */ 
void deleteVector(double *ptr); 

/* Get current date/time, format is YYYY-MM-DD hh:mm:ss */
const string currentDateTime();

