import json
from byzfl.benchmark.evaluate_results import *

with open('config.json', 'r') as f:
    default_config = json.load(f)
    # with open('ref_config.json', 'r') as f:
    #     ref_config = json.load(f)

    path_training_results = default_config["evaluation_and_results"]["results_directory"]
    path_to_plot = path_training_results + "/plot"
    nb_steps = 4000
    use_ref= True
    zoom=True
    path_to_results_ref = "./results/mnist_logreg_full_4"# None #./results/mnist_logreg_iid_2"

    paper_used_plots(path_training_results, path_to_plot, metric='train_loss',
                     use_ref=use_ref, nb_steps_displayed=nb_steps, path_to_results_ref=path_to_results_ref, zoom=zoom)
    paper_used_plots(path_training_results, path_to_plot, metric='test_accuracy',use_ref=use_ref, nb_steps_displayed=nb_steps,path_to_results_ref=path_to_results_ref, zoom=zoom)
    paper_used_plots(path_training_results, path_to_plot, metric='train_accuracy',use_ref=use_ref, nb_steps_displayed=nb_steps,path_to_results_ref=path_to_results_ref, zoom=zoom)
