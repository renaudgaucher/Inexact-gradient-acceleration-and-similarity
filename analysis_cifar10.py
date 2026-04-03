import json
from byzfl.benchmark.evaluate_results import *

with open('config.json', 'r') as f:
    default_config = json.load(f)

    path_training_results = default_config["evaluation_and_results"]["results_directory"]
    path_to_plot = path_training_results + "/plot"

    # Plot test accuracy with scenarios differentiated by 'attack'
    plot_metric_curve(path_training_results, path_to_plot, metric='test_accuracy', vary_dimension='attack')

    # Plot train loss with scenarios differentiated by 'attack'
    plot_metric_curve(path_training_results, path_to_plot, metric='train_loss', vary_dimension='attack', zoom_inset=False)

    # Plot validation accuracy with scenarios differentiated by 'attack'
    plot_metric_curve(path_training_results, path_to_plot, metric='val_accuracy', vary_dimension='attack')
