import json
from byzfl.benchmark.evaluate_results import *

with open('config.json', 'r') as f:
    default_config = json.load(f)

    path_training_results = default_config["evaluation_and_results"]["results_directory"]
    path_to_plot =  "plot/anaylsis_att_def"

    label_dimension =  'training_algorithm' # "agg" 'training_algorithm', 'lr_list', 'momentum_list', 'nb_honest', 'nb_byzantine', 'nb_decl', 'nb_nodes', 'data_dist', 'distribution_parameter', 'pre_agg_names', 'agg', 'attack'
    group_dimension = ("agg","attack")
    use_ref=True
    plot_ref=True
    path_to_results_ref=None
    groups_in_subplots=True
    ylim=1e-4,1 # (None,None)

    plot_metric_curve(path_training_results, path_to_plot, metric='train_loss', group_dimension=group_dimension, label_dimension=label_dimension,
                      zoom_inset=False,ylim=ylim, use_ref=use_ref, plot_ref=plot_ref, path_to_results_ref=path_to_results_ref,
                      groups_in_subplots=groups_in_subplots)

    ylim = (0.9,0.925)
    plot_metric_curve(path_training_results, path_to_plot, metric='test_accuracy', group_dimension=group_dimension, label_dimension=label_dimension,
                          zoom_inset=False,ylim=ylim, use_ref=use_ref, plot_ref=plot_ref, path_to_results_ref=path_to_results_ref,
                          groups_in_subplots=groups_in_subplots)

