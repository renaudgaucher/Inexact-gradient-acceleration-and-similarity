import json
# from byzfl.benchmark.evaluate_results import *
from byzfl.benchmark.evaluate_results_old import *

with open('config.json', 'r') as f:
    default_config = json.load(f)

    path_training_results = default_config["evaluation_and_results"]["results_directory"]
    path_to_plot = "./plot/mnist/per_attack"

    nb_steps = 3000
    # use_ref= True
    use_ref=False
    path_to_results_ref = None 
    plot_ref=True
    paper_used_plots_with_ref_aggregation_comparison(path_training_results, path_to_plot, metric='train_loss', nb_steps_displayed=nb_steps,
                              path_to_results_ref=path_to_results_ref, zoom=use_ref)
    