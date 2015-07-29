----------------------------------------------------------------------------
README file for Baseline
----------------------------------------------------------------------------

Author: Jamie Zhu <jimzhu@GitHub>
Last updated: 2014/11/14.

This package implements a baseline approach for service reliability prediction 
by using the average values of historical data. 

----------------------------------------------------------------------------
Dependencies
----------------------------------------------------------------------------

- Python 2.7 (https://www.python.org)
- numpy 1.8.1 (http://www.scipy.org)
- scipy 0.13.3 (http://www.scipy.org)

----------------------------------------------------------------------------
Contents of this package
----------------------------------------------------------------------------

Baseline/
  - run_rel.py                   - script file for running the experiments on 
                                   reliability QoS data 
  - setup.py                     - setup script file for build c++ modules
  - readme.txt                   - descriptions of this package 
  - src/                         - directory of the source files
      - dataloader.py            - a function to load the dataset (with 
	                               preprocessing)
      - utilities.py             - a script containing a bag of useful utilities
      - evaluator.py             - control execution and results collection of the 
                                   specific algorithm
      - core.py                  - Baseline approach
  - result/                      - directory for storing evaluation results
                                   available metrics: (MAE, NMAE, RMSE, MRE, NPRE)
      - avg_relResult_0.02.txt   - E.g., the reliability prediction result under 
                                   matrix density = 2%
      - [...]                    - many other results

----------------------------------------------------------------------------
Usage of this package
----------------------------------------------------------------------------

For ease of reproducing and compare with other approaches, we provide the 
detailed experimental results with five metrics (MAE, NMAE, RMSE, MRE, NPRE), 
under the "result/" directory, after running the above QoS prediction approach 
on our dataset. E.g.,"result/avg_relResult_0.02.txt" records the evaluation 
results under matrix density = 2%. In particular, each experiment is run for 
20 times and the average result (including std value) is reported. These 
results can be directly used for your research work.

On the other hand, if you want to reproduce our experiments, you can run the 
program with our provided Python scripts "run_rt.py". You can also turn on 
the "parallelMode" in the config area for speedup if you use a multi-core 
computer.

>> python run_rel.py

----------------------------------------------------------------------------
Issues
----------------------------------------------------------------------------

In case of questions or problems, please do not hesitate to report to our 
issue page (https://github.com/wsdream/CARP/issues). We will help ASAP. 
In addition, we will appreciate any contribution to refine and optimize this 
package.

----------------------------------------------------------------------------
Copyright
----------------------------------------------------------------------------

Copyright (c) WS-DREAM@GitHub, CUHK.

Permission is granted for anyone to copy, use, modify, or distribute this 
program and accompanying programs and documents for any purpose, provided 
this copyright notice is retained and prominently displayed, along with a 
note saying that the original programs are available from our web page 
(https://rmblab.github.io). The program is provided as-is, and there are 
no guarantees that it fits your purposes or that it is bug-free. All use 
of these programs is entirely at the user's own risk. For any enquiries, 
please feel free to contact Jamie Zhu <jmzhu AT cse.cuhk.edu.hk>.

