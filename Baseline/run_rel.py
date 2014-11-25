########################################################
# run_rel.py: reliability prediction
# Author: Jamie Zhu <jimzhu@GitHub>
# Created: 2014/2/6
# Last updated: 2014/11/14
# Implemented approach: Baseline
# Evaluation metrics: MAE, NMAE, RMSE, MRE, NPRE
########################################################

import numpy as np
import os, sys, time
import multiprocessing
sys.path.append('src')
from utilities import *
import evaluator
import dataloader
 

#########################################################
# config area
#
para = {'dataType': 'rel', # choose between 'rt' and 'rel'
        'dataPath': '../data/fse13_data.csv',
		'outPath': 'result/',
		'metrics': ['MAE', 'NMAE', 'RMSE', 'MRE', 'NPRE'], # delete where appropriate		
		'density': list(np.arange(0.05, 0.26, 0.05)), # matrix density
		'rounds': 20, # how many runs are performed at each matrix density
        'numEvClus': 7, # number of clusters of environment
        'numUsClus': 7, # number of clusters of users
        'numWsClus': 7, # number of clusters of services
		'saveTimeInfo': False, # whether to keep track of the running time
		'saveLog': True, # whether to save log into file
		'debugMode': False, # whether to record the debug info
        'parallelMode': True # whether to leverage multiprocessing for speedup
		}

initConfig(para)
#########################################################


startTime = time.clock() # start timing
logger.info('==============================================')
logger.info('Baseline approach.')

# load the dataset
dataTensor = dataloader.load(para)
logger.info('Loading data done.')

# run for each density
if para['parallelMode']: # run on multiple processes
    pool = multiprocessing.Pool()
    for density in para['density']:
        pool.apply_async(evaluator.execute, (dataTensor, density, para))
    pool.close()
    pool.join()
else: # run on single processes
    for density in para['density']:
		evaluator.execute(dataTensor, density, para)

logger.info(time.strftime('All done. Total running time: %d-th day - %Hhour - %Mmin - %Ssec.',
         time.gmtime(time.clock() - startTime)))
logger.info('==============================================')
sys.path.remove('src')