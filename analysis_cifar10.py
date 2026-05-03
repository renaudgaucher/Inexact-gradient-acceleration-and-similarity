import json
from byzfl.benchmark.evaluate_results import *

with open('config.json', 'r') as f:
    default_config = json.load(f)

    path_training_results = default_config["evaluation_and_results"]["results_directory"]
    path_to_plot = path_training_results + "/plot"

    # group_dimension, label_dimension = "training_algorithm", "training_algorithm" # 'training_algorithm', 'lr_list', 'momentum_list', 'nb_honest', 'nb_byzantine', 'nb_decl', 'nb_nodes', 'data_dist', 'distribution_parameter', 'pre_agg_names', 'agg', 'attack'
    group_dimension, label_dimension = "attack", "training_algorithm"

    show_lr_mom=True

    # Plot test accuracy with scenarios differentiated by 'attack'
    plot_metric_curve(path_training_results, path_to_plot, metric='test_accuracy', group_dimension=group_dimension, label_dimension=label_dimension, show_lr_mom=show_lr_mom)

    # Plot train loss with scenarios differentiated by 'attack'
    plot_metric_curve(path_training_results, path_to_plot, metric='train_loss',
                       group_dimension=group_dimension, zoom_inset=False, label_dimension=label_dimension,show_lr_mom=show_lr_mom,
                       mv_avg_window=50)

    # Plot validation accuracy with scenarios differentiated by 'attack'
    # plot_metric_curve(path_training_results, path_to_plot, metric='val_accuracy', group_dimension=group_dimension)
