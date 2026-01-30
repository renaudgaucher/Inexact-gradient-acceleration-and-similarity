def make_ref_config(config, steps_multiplier=1, same_algs=True):
    ref_config = config.copy()
    ref_config["benchmark_config"]["nb_steps"] = int(ref_config["benchmark_config"]["nb_steps"] * steps_multiplier)
    ref_config["model"]["weight_decay"] = [0.001] #0.01, 
    ref_config["benchmark_config"]["f"] = [0]
    if not same_algs:
        ref_config["benchmark_config"]["training_algorithm"] = [{
                "name": "DSGD",
                "parameters": {
                    "optimizer_name": "SGD",
                    "momentum": [0.9],
                    "optimizer_parameters": { 
                                "nesterov": True
                                },
                    "learning_rate": [0.05], 
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
                    "learning_rate": [1., 2., 4., 8.],# oldies [1.,2.,4.,8.,16.] [0.1,0.5,1.,2.,4.,8.,16.,32.,64.], #[0.025,0.05,0.1,0.25,0.5],#,0.75,1.0],
                    "learning_rate_decay": 1.0,
                    "milestones": []
                    }
                }]
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