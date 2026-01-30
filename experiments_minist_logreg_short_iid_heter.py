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
        "nb_honest_clients": 20,
        "f": [1,5],
        "size_train_set": 0.8,
        "data_distribution_seed": 0,
        "nb_data_distribution_seeds": 1,
        "data_distribution": [
            {
                "name": "iid",
                "distribution_parameter": [None], 
            },    
            {
                "name": "dirichlet_niid",
                "distribution_parameter": [1.], 
            },
        ],
        "training_algorithm": [
            {
            "name": "DSGD",
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.9],
                "optimizer_parameters": { 
                            "nesterov": True
                            },
                "learning_rate": [0.1], 
                "learning_rate_decay": 1.0,
                "milestones": []
                }
            },
            {
            "name": "moDSGD",
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.],
                "optimizer_parameters": { 
                            },
                "learning_rate": [0.1], 
                "learning_rate_decay": 1.0,
                "milestones": []
                }
            },
            {
            "name": "FedProxyProx",
            "parameters": {
                "optimizer_name": "SGD",
                "momentum": [0.],
                "optimizer_parameters": { 
                            },
                "learning_rate":  [32.,16., 8., 4.,2.,1.], 
                "learning_rate_decay": 1.0,
                "milestones": []
                }
            },
        ],
        "nb_steps": 500,
    },
    "model": {
        "name": "logreg_mnist",
        "dataset_name": "mnist",
        "nb_labels": 10,
        "loss": "NLLLoss",
        "weight_decay": [0.001],  
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
        "batch_size": 0 # full batch
    },
    "attack": [
        {
            "name": "Optimal_InnerProductManipulation",
            "parameters": {}
        },
        {
            "name": "Optimal_ALittleIsEnough",
            "parameters": {}
        },
    ],
    "evaluation_and_results": {
        "evaluation_delta": 2,
        "batch_size_evaluation": 2**6,
        "evaluate_on_test": True,
        "store_per_client_metrics": True,
        "store_models": False,
        "data_folder": "./data",
        "results_directory": "./results/mnist_logreg_iid_2",
    }
}


if __name__ == "__main__":
    with open('config.json', 'w') as f:
        json.dump(default_config, f, indent=4)
    run_benchmark(10)

    with open('ref_config.json', 'w') as f:
        from ref_config import make_ref_config
        ref_config = make_ref_config(default_config, steps_multiplier=1.5)
        json.dump(ref_config, f, indent=4)
    run_benchmark(10, config_name='ref_config.json')