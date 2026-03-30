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
        "nb_honest_clients": 12, # to chose. For instance I guess n=20 + f = 1 or 2 is good
        "f": [0], # to choose. Tune with f=0
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
            "name": "DSGD", ### This is NAG
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.9],# to tune
                "optimizer_parameters": { 
                            "nesterov": True
                            },
                "learning_rate": [1.], # to tune
                "learning_rate_decay": 1.0,
                "milestones": []
                }
            },
            {
            "name": "moDSGD", # This is SGD
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.],
                "optimizer_parameters": { 
                            },
                "learning_rate": [ 1.], # to tune
                "learning_rate_decay": 1.0,
                "milestones": []
                }
            },
            {
            "name": "FedProxyProx", # This is ProxyProx
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.],
                "optimizer_parameters": { 
                            },
                "learning_rate": [32.], # to tune
                "learning_rate_decay": 1.0,
                "milestones": []
                }
            },
        ],
        "nb_steps": 200,
    },
    "model": {
        "name": "cnn_cifar",
        "dataset_name": "cifar10",
        "nb_labels": 10,
        "loss": "NLLLoss",
        "weight_decay": [0.001],   # l2_regularization
    },
    "aggregator": [
        {
            "name": "TrMean",
            "parameters": {}
        }
    ],
    "pre_aggregators": [
        {
            "name": "NNM",
            "parameters": {}
        }
    ],
    "honest_clients": {
        "batch_size": 2**10 # full batch is 0, smaller batch size for easier configuration is good atm
    },
    "attack": [
        {
            "name": "Optimal_InnerProductManipulation",
            "parameters": {}
        },
        # {
        #     "name": "Optimal_ALittleIsEnough",
        #     "parameters": {}
        # },
    ],
    "evaluation_and_results": {
        "evaluation_delta": 100,
        "batch_size_evaluation": 2**6, # to increase for more accuracy
        "evaluate_on_test": True,
        "store_per_client_metrics": True,
        "store_models": False,
        "data_folder": "./data",
        "results_directory": "./results/mnist_cifar10_0",
    }
}


if __name__ == "__main__":
    with open('config.json', 'w') as f:
        json.dump(default_config, f, indent=4)
    run_benchmark(2)
