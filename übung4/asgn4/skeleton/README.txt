
1) Install Anaconda3.

    Instructions: https://www.anaconda.com/distribution
    Note: Use Python3 version.


2) Install PyMaxFlow via ``$pip install PyMaxFlow``

    Note: Make sure you are using the pip command of your Anaconda distribution.
    GitHub: https://github.com/pmneila/PyMaxflow
    Docs: http://pmneila.github.io/PyMaxflow

    If this does not work, it may help to downgrade to Python 3.6 before:
        ``$conda install python=3.6``


3) Install PyTorch
    Instructions: https://pytorch.org/get-started/locally/

    Linux:   $conda install pytorch-cpu torchvision-cpu -c pytorch
    MaxOS:   $conda install pytorch torchvision -c pytorch
    Windows: $conda install pytorch-cpu torchvision-cpu -c pytorch


4) You can try:

      $python run_tests.py

   to see whether your environment is set up correctly.