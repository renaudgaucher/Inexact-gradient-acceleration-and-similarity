#!/bin/bash 

#OAR -q besteffort
#OAR -l host=1/gpu=1, walltime=16:00:00
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# display some information about attributed resources
hostname 
nvidia-smi 
 
# make use of a python torch environment
module load conda
module load cudnn
conda activate renv2
python3 experiments_cifar10.py
