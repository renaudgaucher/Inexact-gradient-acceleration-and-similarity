import json
# from byzfl.benchmark.evaluate_results import *
from byzfl.benchmark.evaluate_results_old import *

with open('config.json', 'r') as f:
    default_config = json.load(f)

    path_training_results = default_config["evaluation_and_results"]["results_directory"]
    path_to_plot = path_training_results + "/plot"



    training_algorithms_comparison_curve(path_to_results=path_training_results, path_to_plot=path_to_plot,
                                         metric='train_loss')
    training_algorithms_comparison_curve(path_to_results=path_training_results, path_to_plot=path_to_plot,
                                         metric='test_accuracy')
    
    # paper_used_plots(path_to_results=path_training_results, path_to_plot=path_to_plot,
    #                  metric='train_loss', nb_steps_displayed=50)
    
    