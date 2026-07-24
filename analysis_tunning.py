import json
from byzfl.benchmark.evaluate_results_old import *

with open('config.json', 'r') as f:
    default_config = json.load(f)
    # with open('ref_config.json', 'r') as f:
    #     ref_config = json.load(f)

    path_training_results = default_config["evaluation_and_results"]["results_directory"]
    path_to_plot = "./plot/cifar/acc"
    nb_steps = None #5000
    use_ref= False# True
    zoom= False#True
    path_to_results_ref = None 

    paper_used_plots(path_training_results, path_to_plot, use_ref=use_ref,
                     metric='train_loss', nb_steps_displayed=nb_steps,
                     path_to_results_ref = path_to_results_ref, zoom=zoom
                     )
