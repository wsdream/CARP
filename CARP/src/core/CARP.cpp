/********************************************************
 * CARP.cpp
 * C++ implements on CARP
 * Author: Jamie Zhu <jimzhu@GitHub>
 * Created: 2014/5/6
 * Last updated: 2014/11/03
********************************************************/

#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include "CARP.h"
#include <vector>
#include <ctime>
#include <algorithm>

using namespace std;

/// note that simMatrixData is the output of this function
void getSimMatrix(double *invocMatrixData, double *simMatrixData, int numTimeSlice, 
    int numInvocations)
{  
    // --- transfer the 1D pointer to 2D array pointer
    double **invocMatrix = vector2Matrix(invocMatrixData, numTimeSlice, numInvocations);
    double **simMatrix = vector2Matrix(simMatrixData, numTimeSlice, numTimeSlice);

    // --- create vectors and matrices
    double *meanVec = createVector(numTimeSlice);

    // --- get average values
    int i, j; 
    for (i = 0; i < numTimeSlice; i++) {
        double avg = 0;
        int count = 0;
        for (j = 0; j < numInvocations; j++) {
            if (invocMatrix[i][j] > 0) {
                avg += invocMatrix[i][j];
                count++;
            }
        }
        if (count == 0) continue;
        meanVec[i] = avg / count;
    }

    // --- get similarity matrix
    for (i = 0; i < numTimeSlice; i++) {    
        for (j = i + 1; j < numTimeSlice; j++) {
            if(meanVec[i] == 0 || meanVec[j] == 0) continue;
            double pccValue = getPCC(invocMatrix[i], invocMatrix[j], 
                meanVec[i], meanVec[j], numInvocations);
            simMatrix[i][j] = pccValue;
            simMatrix[j][i] = pccValue;
        }
    }

    deleteVector(meanVec);
    delete ((char*) invocMatrix);
    delete ((char*) simMatrix);
}


/// note that predData is the output of this function
void CARP(double *removedData, double *predData, int numUser, int numService, 
	int numContext, int dim, double lmda, int maxIter, bool debugMode, 
	double *Udata, double *Sdata, double *Cdata)
{	
	// --- transfer the 1D pointer to 2D/3D array pointer
    double ***R = vector2Tensor(removedData, numUser, numService, numContext);
    double ***R_hat = vector2Tensor(predData, numUser, numService, numContext);
    double **U = vector2Matrix(Udata, numUser, dim);
    double **S = vector2Matrix(Sdata, numService, dim);
    double ***C = vector2Tensor(Cdata, dim, dim, numContext);

    // iteration
    double t1, t2, up, down;
    for (int iter = 1; iter <= maxIter; iter++) {
        // update R_hat
        updateR_hat(false, R, R_hat, U, S, C, numUser, numService, numContext, dim);
        
        // log the debug info
        cout.setf(ios::fixed);
        if (debugMode) {
            pdd loss = lossFunction(R, R_hat, U, S, C, numUser, numService, numContext, 
                dim, lmda);
            cout << currentDateTime() << ": ";
            cout << "iter = " << iter << ", lossValue = " << loss.first + loss.second 
                << ", cost = " << loss.first << ", reg = " << loss.second << endl;
        }

        int i, j, k, l, g;
         // update U
         for (i = 0; i < numUser; i++) {          
            for (l = 0; l < dim; l++) {
                up = 0, down = 0;
                for (k = 0; k < numContext; k++) {
                    for (g = 0; g < dim; g++) {
                        t1 = 0, t2 = 0;
                        for (j = 0; j < numService; j++) {
                            t1 += R[i][j][k] * S[j][g];
                            t2 += R_hat[i][j][k] * S[j][g];
                        }
                        up += t1 * C[l][g][k];
                        down += t2 * C[l][g][k];
                    }
                }
                U[i][l] *= sqrt(up / (down + lmda * U[i][l] + eps));
            }                      
        }

        // update R_hat
        updateR_hat(false, R, R_hat, U, S, C, numUser, numService, numContext, dim);

        // update S
        for (j = 0; j < numService; j++) {          
            for (l = 0; l < dim; l++) {
                up = 0, down = 0;
                for (k = 0; k < numContext; k++) {
                    for (g = 0; g < dim; g++) {
                        t1 = 0, t2 = 0;
                        for (i = 0; i < numUser; i++) {
                            t1 += R[i][j][k] * U[i][g];
                            t2 += R_hat[i][j][k] * U[i][g];
                        }
                        up += t1 * C[g][l][k];
                        down += t2 * C[g][l][k];
                    }
                }
                S[j][l] *= sqrt(up / (down + lmda * S[j][l] + eps));
            }                      
        }

        // update R_hat
        updateR_hat(false, R, R_hat, U, S, C, numUser, numService, numContext, dim);

        // update C(k)
        for (k = 0; k < numContext; k++) {          
            for (l = 0; l < dim; l++) {
                for (g = 0; g < dim; g++) {
                    up = 0, down = 0;
                    for (j = 0; j < numService; j++) {
                        t1 = 0, t2 = 0;  
                        for (i = 0; i < numUser; i++) {
                            t1 += U[i][g] * R[i][j][k];
                            t2 += U[i][g] * R_hat[i][j][k];
                        }
                        up += t1 * S[j][l];
                        down += t2 * S[j][l];
                    }
                    C[g][l][k] *= sqrt(up / (down + lmda *  C[g][l][k] + eps));
                }             
            }                      
        }
    }

    // update R_hat
    updateR_hat(true, R, R_hat, U, S, C, numUser, numService, numContext, dim);

    // clear
    delete ((char*) U);
    delete ((char*) S);
    delete ((char*) C);
    delete ((char*) R);
    delete ((char*) R_hat);
}


double getPCC(double *uA, double *uB, double meanA, double meanB, int numInvocations) 
{
    vector<int> commonIndex;
    int i;
    for (i = 0; i < numInvocations; i++) {
            if(uA[i] > 0 && uB[i] > 0) {
                commonIndex.push_back(i);
            }
    }

    // no common rate items. 
    if(commonIndex.size() < 2) return 0;

    double upperAll = 0;
    double downAllA = 0;
    double downAllB = 0;
    for (i = 0; i < commonIndex.size(); i++) {
        int key = commonIndex[i];
        double valueA = uA[key];
        double valueB = uB[key];

        double tempA = valueA - meanA;
        double tempB = valueB - meanB;

        upperAll += tempA * tempB;
        downAllA += tempA * tempA;
        downAllB += tempB * tempB;
    }  
    double downValue = sqrt(downAllA * downAllB);

    if(downValue == 0) return 0;
    double pcc = upperAll / downValue;

    return pcc;
}


void updateR_hat(bool flag, double ***R, double ***R_hat, double **U, double **S, 
    double ***C, int numUser, int numService, int numContext, int dim)
{
    for (int i = 0; i < numUser; i++) {
        for (int j = 0; j < numService; j++) {
            for (int k = 0; k < numContext; k++) {
                if (flag == true || R[i][j][k] > 0) {
                    double result = 0;
                    for (int l = 0; l < dim; l++) {
                        double tmp = 0;
                        for (int g = 0; g < dim; g++) {
                            tmp += U[i][g] * C[g][l][k];
                        }
                        result += tmp * S[j][l];
                    }
                    R_hat[i][j][k] = result;
                }
            }
        }
    }
}


inline double sqr(double x) {return x * x;}

pdd lossFunction(double ***R, double ***R_hat, double **U, double **S, double ***C, 
    int numUser, int numService, int numContext, int dim, double lmda)
{
    double reg = 0, cost = 0;
    
    for (int l = 0; l < dim; l++) {
        for (int i = 0; i < numUser; i++) {
            reg += sqr(U[i][l]);
        }
        for (int j = 0; j < numService; j++) {
            reg += sqr(S[j][l]);
        }
        for (int k = 0; k < numContext; k++) {
            for (int g = 0; g < dim; g++)
            reg += sqr(C[g][l][k]);
        }
    }
    reg *= lmda;

    for (int i = 0; i < numUser; i++) {
        for (int j = 0; j < numService; j++) {
            for (int k = 0 ; k < numContext; k++) {
                if (R[i][j][k] > 0) {
                    cost += sqr(R[i][j][k] - R_hat[i][j][k]);
                }
            }
        }
    }
    
    return pdd(cost / 2, reg / 2);
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


double *createVector(int size) 
{
    double *vec = new double[size];
    memset(vec, 0, size * sizeof(double)); // Initialization
    return vec;
}


void deleteVector(double *ptr) {
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

