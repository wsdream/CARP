##CARP

This repository maintains the source code of our implementation on CARP, context-aware reliability prediction of black-box services.

Read more information: [[Paper](http://arxiv.org/abs/1503.00102)][[Project page](http://wsdream.github.io/CARP)]


###Related Links
- WSRec paper list: http://wsdream.github.io/WSRec/paperlist

- WS-DREAM datasets: http://wsdream.github.io/dataset


###Dependencies
- Python 2.7 (https://www.python.org)
- Cython 0.20.1 (http://cython.org)
- numpy 1.8.1 (http://www.scipy.org)
- scipy 0.13.3 (http://www.scipy.org)

The benchmarks are implemented with a combination of Python and C++. The framework is built on Python for simplicity, and the core functions of each algorithm are written in C++ for efficiency consideration. To achieve so, [Cython](http://cython.org/ "Cython's Web page") (a language to write C/C++ extensions for Python) has been employed to compile the C++ extensions to Python-compatible modules. In our repository, Cython (with version 0.20.1) has been set as a sub-module in the "externals/Cython" directory.

>Note: Our code is directly executable on Linux platform. Re-compilation with Cython is required to execute them on Windows platform: See [how to run on Windows](https://github.com/wsdream/WSRec#usage). 


###Feedback
For bugs and feedback, please post to [our issue page](https://github.com/wsdream/CARP/issues). For any other enquires, please drop an email to our team (wsdream.maillist@gmail.com).


###Copyright &copy;
Permission is granted for anyone to copy, use, modify, or distribute this program and accompanying programs and documents for any purpose, provided this copyright notice is retained and prominently displayed, along with a note saying that the original programs are available from our web page (https://wsdream.github.io). The program is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. All use of these programs is entirely at the user's own risk.

