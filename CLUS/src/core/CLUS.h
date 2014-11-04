/********************************************************
 * CLUS.h: header file of CLUS.cpp
 * Author: Jamie Zhu <jimzhu@GitHub>
 * Created: 2014/5/6
 * Last updated: 2014/7/15
********************************************************/

#include <algorithm>
#include <vector>
using namespace std;


/* Perform the core approach of CLUS */
void CLUS_core(double *removedData, double *predData, int numUser, int numService, 
	int numTimeSlice, vector<int> attrEv, vector<int> attrUs, vector<int> attrWs, 
    vector<vector<int> > clusterEv, vector<vector<int> > clusterUs, 
    vector<vector<int> > clusterWs, bool debugMode);

/* Transform a vector into a matrix */ 
double **vector2Matrix(double *vector, int row, int col);

/* Transform a vector into a 3D tensor */ 
double ***vector2Tensor(double *vector, int row, int col, int height);
bool ***vector2Tensor(bool *vector, int row, int col, int height);

/* Allocate memory for a 2D array */
double **createMatrix(int row, int col);

/* Free memory for a 2D array */ 
void delete2DMatrix(double **ptr); 

/* Get current date/time, format is YYYY-MM-DD hh:mm:ss */
const string currentDateTime();

