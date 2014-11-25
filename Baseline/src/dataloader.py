########################################################
# dataloader.py
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/11/14
########################################################

import numpy as np 
from utilities import *


########################################################
# Function to load the dataset
#
def load(para):
	datafile = para['dataPath']
	logger.info('Loading data: %s'%datafile)
	if para['dataType'] == 'rel':
		dataTensor = np.zeros((50, 49, 7))
		for i in range(7):
			data = np.genfromtxt(datafile, comments='$', delimiter=',')
			dataTensor[:, :, i] = data[(2 + 52 * i):(2 + 52 * i + 50), 1:]
	else: # for 'rt'
		dataTensor = -1 * np.ones((420, 1000, 480))
		with open(datafile) as lines:
			for line in lines:
				data = line.strip().split('\t')		
				rt = float(data[3])
				if rt > 0:
					dataTensor[int(data[1]), int(data[2]), int(data[0])] = rt

	return dataTensor 
########################################################

