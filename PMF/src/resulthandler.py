########################################################
# resulthandler.py: get the average values of the results
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/11/14
########################################################

import numpy as np
import linecache
import os, sys, time
 

########################################################
# Get the average statistics
#
def averageStats(para):
	if para['dataType'] == 'rel':
		timeSlice = 7
	lineIdToExtract = 2
	resultFolder = para['outPath'].split('/')[0] + '/'
			
	for den in para['density']:
		result = []
		for timeslice in range(timeSlice):
			inputfile = para['outPath'] + '%02d_%sResult_%.2f.txt'\
				%(timeslice + 1, para['dataType'], den)
			data = linecache.getline(inputfile, lineIdToExtract).strip().split('\t')
			metrics = [float(x) for x in data[1:]]
			result.append(metrics)
		outfile = resultFolder + 'avg_%sResult_%.2f.txt'%(para['dataType'], den)
		saveAvgResult(outfile, np.array(result), para)
		avgResult = np.average(result, axis = 0)
		print avgResult
########################################################


########################################################
# Save the average results into file
#
def saveAvgResult(outfile, result, para):
    fileID = open(outfile, 'w')
    fileID.write('Metric: ')
    for metric in para['metrics']:
        fileID.write('| %s\t'%metric)
    avgResult = np.average(result, axis = 0)         
    fileID.write('\nAvg:\t')
    np.savetxt(fileID, np.matrix(avgResult), fmt='%.4f', delimiter='\t')
    stdResult = np.std(result, axis = 0)
    fileID.write('Std:\t')
    np.savetxt(fileID, np.matrix(stdResult), fmt='%.4f', delimiter='\t')
    fileID.write('\n==========================================\n')
    fileID.write('Detailed results for %d slices:\n'%result.shape[0])
    np.savetxt(fileID, result, fmt='%.4f', delimiter='\t')     
    fileID.close()
########################################################
