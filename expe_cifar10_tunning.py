import sys
import os
import json
import torch

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from byzfl import run_benchmark


default_config = {
    "benchmark_config": {
        "dtype": "64",
        "device": "cuda",# if torch.cuda.is_available() else "cpu",
        "training_seed": 0,
        "nb_training_seeds": 1,
        "nb_honest_clients": 40, # to chose. For instance I guess n=20 + f = 1 or 2 is good
        "f": [0],#, 2], # to choose. Tune with f=0
        "size_train_set": 0.8,
        "data_distribution_seed": 0,
        "nb_data_distribution_seeds": 1, 
        "data_distribution": [
            {
                "name": "dirichlet_niid",
                "distribution_parameter": [5.], # ok-ish, perhaps 1. is better, but harder
            },
        ],
        "training_algorithm": [ 
            {
            "name": "moDSGD", # This is SGD
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.],
                "optimizer_parameters": { 
                            },
                "learning_rate": [5.,2.,1.,0.5,0.1], #[0.5, 0.1], # tuned for cifar10
                "milestones": []
                }
            },
            # {
            # "name": "AccExtraGrad",
            # "parameters": {
            #     "optimizer_name": "SGD",
            #     "optimizer_parameters": {
            #                 },
            #     "learning_rate":  [0.05], 
            #     "momentum":[0.01], # Not a true momentum term: corresponds to 'gamma', which in theory,
            #     # corresponds to the strong convexity. Should always be smaller than 1/lr.
            #     # "tau_factor": 1, (multiplies the tau - i.e. the averaging factor)
            #     # "fast_lr_factor": 1., (multiplies the large step size)
            #     }
            # },
            # {
            # "name": "FedProxyProx",
            # "parameters": {
            #     "optimizer_name": "SGD",
            #     "momentum": [0.],
            #     "optimizer_parameters": { 
            #                 },
            #     "learning_rate":  [1.], #10.,
            #     "prox_optimizaer_name": "SGD",
            #     "prox_optimizer_params": {"learning_rate" : 0.05,
            #                               "security_factor" : 10},
                
            #     }
            # },
            # {
            # "name": "AccExtraGradProx",
            # "parameters": {
            #     "optimizer_name": "SGD",
            #     "optimizer_parameters": { 
            #                 },
            #     "learning_rate":  [0.2],#[0.5,0.2], #5.,1. not converging
            #     "momentum":[0.005],
            #     "tau_factor": 1,
            #     "fast_lr_factor": 1, 
            #     "prox_optimizaer_name": "SGD",
            #     "prox_optimizer_params": {"learning_rate" : 0.05,
            #                               "security_factor" : 10},
            #     }
            # },
        ],
        "nb_steps": 100,
    },
    "model": {
        "name": "cnn_cifar",
        "dataset_name": "cifar10",
        "nb_labels": 10,
        "loss": "NLLLoss",
        "weight_decay": [0.01],   # l2_regularization
    },
    "aggregator": [
        {
            "name": "Huber",
            "parameters": {}
        }
    ],
    "pre_aggregators": [
        # {
        #     "name": "NNM",
        #     "parameters": {}
        # }
    ],
    "honest_clients": {
        "batch_size": 0 # full batch is 0, smaller batch size for easier configuration is good atm
    },
    "attack": [
        {
            "name": "Optimal_InnerProductManipulation",
            "parameters": {}
        },
        #{
        #    "name": "Optimal_ALittleIsEnough",
        #    "parameters": {}
        #},
        # {
        #     "name": "Gaussian",
        #     "parameters": {}
        # },
        # {
        #     "name": "SignFlipping",
        #     "parameters": {}
        # },
    ],
    "evaluation_and_results": {
        "evaluation_delta": 100,
        "batch_size_evaluation": 2**8, # to increase for more accuracy
        "evaluate_on_test": True,
        "store_per_client_metrics": True,
        "store_models": False,
        "cache_test": True, # to speed up evaluation
        "cache_train": True, # to speed up test
        "data_folder":  "./data",# /tmp
        "results_directory": "./results/cifar10",
        }
}


if __name__ == "__main__":
    with open('config.json', 'w') as f:
        json.dump(default_config, f, indent=4)
    run_benchmark(3)
