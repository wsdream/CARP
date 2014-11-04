/********************************************************
 * CLUS.cpp
 * C++ implements on CLUS
 * Author: Yuwen Xiong <Orpine@GitHub>
 * Created: 2014/7/27
 * Last updated: 2014/7/27
********************************************************/

#include <iostream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <vector>
#include <ctime>
#include <algorithm>
#include "CLUS.h"
using namespace std;

const double eps = 1e-8;

/// note that predData is the output of this function
void CLUS_core(double *removedData, double *predData, int numUser, int numService, 
	int numTimeSlice, vector<int> attrEv, vector<int> attrUs, vector<int> attrWs, 
    vector<vector<int> > clusterEv, vector<vector<int> > clusterUs, 
    vector<vector<int> > clusterWs, bool debugMode)
{	
	// --- transfer the 1D pointer to 2D/3D array pointer
    double ***removedTensor = vector2Tensor(removedData, numUser, numService, numTimeSlice);
    double ***predTensor = vector2Tensor(predData, numUser, numService, numTimeSlice);

    // alias
    double ***Y = removedTensor;
    double ***Y_hat = predTensor;
    
    // --- temporal variables
    bool *tvis = new bool[clusterUs.size() * clusterWs.size() * clusterEv.size()];
    double *tf = new double[clusterUs.size() * clusterWs.size() * clusterEv.size()];
    memset(tvis, false, clusterUs.size() * clusterWs.size() * clusterEv.size());
    
    bool ***vis = vector2Tensor(tvis, clusterUs.size(), clusterWs.size(), clusterEv.size());
    double ***f = vector2Tensor(tf, clusterUs.size(), clusterWs.size(), clusterEv.size());
    
    
    for (int i = 0; i < numUser; ++i) {
    	for (int j = 0; j < numService; ++j) {
    		for (int k = 0; k < numTimeSlice; ++k) {
                if (fabs(Y[i][j][k]) > eps) {
                    Y_hat[i][j][k] = Y[i][j][k];
                    continue;
                }

                // use the historical values in the same context condition for prediction
    			int cnt = 0;
    			double tot = 0;
                for (int l = 0; l < clusterEv[attrEv[k]].size(); l++) {
    				tot += Y[i][j][clusterEv[attrEv[k]][l]];
    				cnt += (fabs(Y[i][j][clusterEv[attrEv[k]][l]]) > eps);
    			}
    			if (cnt != 0) {
    				Y_hat[i][j][k] = tot / cnt;
    				continue;
    			}
                
                // query the hash space f for prediction
                if (vis[attrUs[i]][attrWs[j]][attrEv[k]]) {
                    Y_hat[i][j][k] = f[attrUs[i]][attrWs[j]][attrEv[k]];
                    continue;
                }

                // compute the hash space f
    			cnt = 0;
                tot = 0;
                for (int x = 0; x < clusterUs[attrUs[i]].size(); x++)
    			{
                    for (int y = 0; y < clusterWs[attrWs[j]].size(); y++)
                    {
                        for (int z = 0; z < clusterEv[attrEv[k]].size(); z++)
    					{
    						double Y_entry = Y[clusterUs[attrUs[i]][x]][clusterWs[attrWs[j]][y]]
    							[clusterEv[attrEv[k]][z]];
    						tot += Y_entry;
    						cnt += (fabs(Y_entry) > eps);
    					}
    				}
    			}
                if (cnt != 0) {
                    Y_hat[i][j][k] = tot / cnt;
                    f[attrUs[i]][attrWs[j]][attrEv[k]] = Y_hat[i][j][k];
                    vis[attrUs[i]][attrWs[j]][attrEv[k]] = true;
                    continue;
                }

                // compute user tMean as the prediction value
                cnt = 0;
                tot = 0;
                for (int z = 0; z < numTimeSlice; z++) {
                    tot += Y[i][j][z];
                    cnt += (fabs(Y[i][j][z]) > eps);
                }
                if (cnt != 0) {
                    Y_hat[i][j][k] = tot / cnt;
                    continue;
                }

                // compute user iMean as the prediction value
                cnt = 0;
                tot = 0;
                for (int x = 0; x < numUser; x++) {
                    for (int z = 0; z < numTimeSlice; z++) {
                        tot += Y[x][j][z];
                        cnt += (fabs(Y[x][j][z]) > eps);
                    }
                }
                if (cnt != 0) {
                    Y_hat[i][j][k] = tot / cnt;
                    continue;
                }

                // compute user uMean as the prediction value
                cnt = 0;
                tot = 0;
                for (int y = 0; y < numService; y++) {
                    for (int z = 0; z < numTimeSlice; z++) {
                        tot += Y[i][y][z];
                        cnt += (fabs(Y[i][y][z]) > eps);
                    }
                }
                if (cnt != 0) {
                    Y_hat[i][j][k] = tot / cnt;
                    continue;
                }

                // compute user overall mean as the prediction value
                cnt = 0;
                tot = 0;
                for (int x = 0; x < numUser; x++) {
                    for (int y = 0; y < numService; y++) {
                        for (int z = 0; z < numTimeSlice; z++) {
                            tot += Y[x][y][z];
                            cnt += (fabs(Y[x][y][z]) > eps);
                        }
                    }
                }
                Y_hat[i][j][k] = tot / (cnt + eps);
    		}
    	}
    }  
}


double **vector2Matrix(double *vector, int row, int col)  
{
	double **matrix = new double *[row];
	if (!matrix) {
		cout << "Memory allocation failed in vector2Matrix." << endl;
		return NULL;
	}

	int i;
	for (i = 0; i < row; i++) {
		matrix[i] = vector + i * col;  
	}
	return matrix;
}


double ***vector2Tensor(double *vector, int row, int col, int height)
{
	double ***tensor = new double **[row];
	if (!tensor) {
		cout << "Memory allocation failed in vector2Tensor." << endl;
		return NULL;
	}

	int i, j;
	for (i = 0; i < row; i++) {
		tensor[i] = new double *[col];
		if (!tensor[i]) {
			cout << "Memory allocation failed in vector2Tensor." << endl;
			return NULL;
		}

		for (j = 0; j < col; j++) {
			tensor[i][j] = vector + i * col * height + j * height;
		}
	}

	return tensor;
}


bool ***vector2Tensor(bool *vector, int row, int col, int height)
{
    bool ***tensor = new bool **[row];
    if (!tensor) {
        cout << "Memory allocation failed in vector2Tensor." << endl;
        return NULL;
    }
    
    int i, j;
    for (i = 0; i < row; i++) {
        tensor[i] = new bool *[col];
        if (!tensor[i]) {
            cout << "Memory allocation failed in vector2Tensor." << endl;
            return NULL;
        }
        
        for (j = 0; j < col; j++) {
            tensor[i][j] = vector + i * col * height + j * height;
        }
    }
    
    return tensor;
}


double **createMatrix(int row, int col) 
{
    double **matrix = new double *[row];
    matrix[0] = new double[row * col];
    memset(matrix[0], 0, row * col * sizeof(double)); // Initialization
    int i;
    for (i = 1; i < row; i++) {
    	matrix[i] = matrix[i - 1] + col;
    }
    return matrix;
}


void delete2DMatrix(double **ptr) {
	delete ptr[0];
	delete ptr;
}


const string currentDateTime() 
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}

