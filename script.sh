#!/bin/bash 

#OAR -q p1
#OAR -l host=1/gpu=1, walltime=16:00:00
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 
#OAR -n ByzFL

# display some information about attributed resources
hostname 
nvidia-smi 
 
# make use of a python torch environment
module load conda
module load cudnn
module load cuda
conda activate renv2
python3 expe_cifar10_tunning.py
