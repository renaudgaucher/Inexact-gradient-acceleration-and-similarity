# From Inexact Gradients to Byzantine Robustness: Acceleration and Optimization under Similarity

This code is based on the Byzfl librairy (https://byzfl.epfl.ch).

To install this project, just go the project folder and use

    > pip install .

To reproduce the experiments showed in our paper, please iteratively run

    > python experiments_tunning.py
    > python analysis_tunning.py
    > python experiments_mnist.py
    > python analysis_mnist.py
    > python analysis_short.py

the plots should be available in the folder 'results/mnist'.

All the experiments can run on a personnal laptop without GPU, even though it can takes up to a few days on slow machines. When running experiments, the number of parallel jobs can be ajusted by modfying the #job parameter in experiments*.py files; 

    > run_benchmark(#job)
