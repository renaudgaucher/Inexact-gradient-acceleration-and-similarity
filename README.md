# From Inexact Gradients to Byzantine Robustness: Acceleration and Optimization under Similarity

This code is based on the Byzfl librairy (https://byzfl.epfl.ch).
Main changes are:
- modifications to run the PIGS and Extragradients algorithms
    modifications of clients.py to design a ProxClient that handles proximal local updates
    modifications of train.py to be able to select PIGS or the extragradient (with or without local proximal updates) method
    many small modifications which are harder to track (e.g. how torch optimizer are stored and used in clients)
- modifications to add a Huber Aggregator and a ClippedMixing pre-aggregator
- modification in the data loading 
- Attempt at systematizing the learning rate selection method, by first selecting it in the Byzantine-free setting, and then using it to benchmark the Byzantine resilient case. 

___

To install this project, go the project folder and use

    > pip install .

___

To run the sweep across proportion of Byzantine clients and heterogeneity proceed with the following:

1. Learning rate search: First, we select the learning rate by looking at the byzantine-free case.
    a. run the expe_mnist_tune_gd.py. Inspect by running analysis_alie_tunning.py (selecting only the gd algorithm)
    the selected lr could be the one with the lowest loss, but with highly jagged path, so decreasing it a bit could be necessary, this can be done using the GD_LR_OVERRIDES parameter e.g. expe_mnist_tune_acc.py
    b. then tune the lr of the other methods based on the one of GD, using expe_mnist_tune_acc.py (inspecting lr with analysis_alie_tunning.py)
    c. Finally, all jagged learning rates can be forced manually using LR_OVERRIDES in expe_mnist_prod.py
2. Run the experiments with expe_mnist_prod.py, analyze them using analysis_mnist_alie_prod.py and analysis_mnist_alie_table.py

This produces experiments only with the ALIE attack.

To compare the efficiency of the different attacks on the different accelerated or not methods, 
1. run expe_mnist_att_def.py 
2. run analysis_mnist_att_def.py

___

results are in the folder "results", plots in "plot"
