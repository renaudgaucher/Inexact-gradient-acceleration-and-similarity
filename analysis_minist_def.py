import json
from byzfl.benchmark.evaluate_results import *

with open('config.json', 'r') as f:
    default_config = json.load(f)

    path_training_results = default_config["evaluation_and_results"]["results_directory"]
    path_to_plot = path_training_results + "/plot"

    vary_dimension = "agg" # 'training_algorithm', 'lr_list', 'momentum_list', 'nb_honest', 'nb_byzantine', 'nb_decl', 'nb_nodes', 'data_dist', 'distribution_parameter', 'pre_agg_names', 'agg', 'attack'
    title_dimension = "attack"
    # Plot test accuracy with scenarios differentiated by 'attack'
    plot_metric_curve(path_training_results, path_to_plot, metric='test_accuracy', vary_dimension=vary_dimension, title_dimension=title_dimension)

    # Plot train loss with scenarios differentiated by 'attack'
    plot_metric_curve(path_training_results, path_to_plot, metric='train_loss', vary_dimension=vary_dimension, zoom_inset=False, title_dimension=title_dimension)

    # Plot validation accuracy with scenarios differentiated by 'attack'
    # plot_metric_curve(path_training_results, path_to_plot, metric='val_accuracy', vary_dimension=vary_dimension)
