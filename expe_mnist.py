import sys
import os
import json

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from byzfl import run_benchmark

default_config = {
    "benchmark_config": {
        "dtype": "float64",
        "device": "cpu",
        "training_seed": 0,
        "nb_training_seeds": 1,
        "nb_honest_clients": 40,
        "f": [1],#,[0,1,5],#,1],
        "size_train_set": 0.8,
        "data_distribution_seed": 0,
        "nb_data_distribution_seeds": 1,
        "data_distribution": [
            {
                "name": "dirichlet_niid",
                "distribution_parameter": [5.],#[1.,5.],
            },
        ],
        "training_algorithm": [
            {
            "name": "moDSGD", # plain GD
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.],
                "optimizer_parameters": { 
                            },
                "learning_rate": [0.1], 
                }
            },
            {
            "name": "FedProxyProx", # PIGS
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.],
                "optimizer_parameters": { 
                            },
                "learning_rate":  [2.], #10.,
                "prox_optimizaer_name": "SGD",
                "prox_optimizer_params": {"learning_rate" : 0.05,
                                          "security_factor" : 10},
                
                }
            },
            {
            "name": "AccExtraGradProx", 
            "parameters": {
                "optimizer_name": "SGD",
                "optimizer_parameters": { 
                            },
                "learning_rate":  [0.5],#[0.5,0.2], #5.,1. not converging
                "momentum":[0.01], # should be set as the weight_decay, or anything less than 1/lr
                "tau_factor": 1,
                "fast_lr_factor": 1, 
                "prox_optimizer_name": "SGD",
                "prox_optimizer_params": {"learning_rate" : 0.05,
                                          "security_factor" : 10},
                }
            },
            {
            "name": "AccExtraGrad",
            "parameters": {
                "optimizer_name": "SGD",
                "optimizer_parameters": {
                            },
                "learning_rate":  [0.1], 
                "momentum":[0.01],  # should be set as the weight_decay
                "tau_factor": 1,
                "fast_lr_factor": 1., 
                }
            },
            
            # {
            # "name": "DSGD",
            # "parameters": {
            #     "optimizer_name": "SGD",
            #     "momentum": [0.9],
            #     "optimizer_parameters": { 
            #                 "nesterov": True
            #                 },
            #     "learning_rate": [0.1], 
            #     }
            # },
        ],
        "nb_steps": 5000,
    },
    "model": {
        "name": "logreg_mnist",
        "dataset_name": "mnist",
        "nb_labels": 10,
        "loss": "NLLLoss",
        "weight_decay": [0.01],
    },
    "aggregator": [
        {
            "name": "Huber",
            "parameters": {}
        },
        # {
        #     "name": "TrMean",
        #     "parameters": {}
        # },
        # # {
        # #     "name": "SMEA",
        # #     "parameters": {}
        # # },
        # {
        #     "name": "Krum",
        #     "parameters": {}
        # },
        # {
        #      "name": "CAF",
        #      "parameters": {}
        # },
    ],
    "pre_aggregators": [],
    "honest_clients": {
        "batch_size": 0 # full batch
    },
    "attack": [
        # {
        #     "name": "Optimal_InnerProductManipulation",
        #     "parameters": {}
        # },
        {
            "name": "Optimal_ALittleIsEnough",
            "parameters": {}
        },
        # {
        #     "name": "Gaussian",
        #     "parameters": {}
        # },
        # {
        #     "name": "LabelFlipping",
        #     "parameters": {}
        # },
    ],
    "evaluation_and_results": {
        "evaluation_delta": 100,
        "batch_size_evaluation": 2**6,
        "evaluate_on_test": True,
        "store_per_client_metrics": True,
        "store_models": False,
        "data_folder": "./data",
        "results_directory": "./results/mnist_full",#./results/tunning_acceleration_params_as_theory",
    }
}


if __name__ == "__main__":
    with open('config.json', 'w') as f:
        json.dump(default_config, f, indent=4)
    run_benchmark(8)

    with open('ref_config.json', 'w') as f:
        from ref_config import make_ref_config
        ref_config = make_ref_config(default_config, steps_multiplier=1.)
        json.dump(ref_config, f, indent=4)
    run_benchmark(10, config_name='ref_config.json')