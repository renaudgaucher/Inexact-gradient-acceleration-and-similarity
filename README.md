# From Inexact Gradients to Byzantine Robustness: Acceleration and Optimization under Similarity

This code is based on the Byzfl librairy (https://byzfl.epfl.ch).

To install this project, just go the project folder and use

    > pip install .

To reproduce the experiments showed in our paper, please iteratively run

    > python experiments_minist_logreg_short_iid_heter.py
    > python analysis_short.py
    > python experiments_mnist_logreg_longrun.py
    > python analysis_long.py

the plots should be available in the folder 'results'.
