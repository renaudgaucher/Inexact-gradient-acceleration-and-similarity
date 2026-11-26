def make_ref_config(config, steps_multiplier=1, same_algs=False):
    ref_config = config.copy()
    ref_config["benchmark_config"]["nb_steps"] = int(ref_config["benchmark_config"]["nb_steps"] * steps_multiplier)
    ref_config["benchmark_config"]["f"] = [0]
    if not same_algs:
        ref_config["benchmark_config"]["training_algorithm"] = [
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
                # {
                # "name": "FedProxyProx",
                # "parameters": {
                #     "optimizer_name": "SGD",
                #     "momentum": [0.],
                #     "optimizer_parameters": { 
                #                 },
                #     "learning_rate": [1., 2., 4., 8.],
                #     "learning_rate_decay": 1.0,
                #     "milestones": []
                #     }
                # }
                ]
    ref_config["aggregator"] = [
        {
            "name": "Average",
            "parameters": {}
        }
    ]
    ref_config["pre_aggregators"] = []
    ref_config["attack"] = [{"name": "NoAttack",
            "parameters": {}}]
    return ref_config