import math
import json
import os

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes


size=24
size_legend=24
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': 'Roman',
    'font.weight':'bold',
    #'legend.fontweight':'bold',
    'text.usetex': True,
    'pgf.rcfonts': False,
    "axes.grid" : True,
    'font.size': size,
    'axes.labelsize':size,
    'axes.titlesize':size,
    'figure.titlesize':size,
    'xtick.labelsize':size,
    'ytick.labelsize':size,
    'legend.fontsize':size_legend
})
colors = sns.color_palette('colorblind') 
markers = ['o', 'v', 's', '*', 'd', '^','<']
dashstyle = ['-', '--', '-.', ':', ':','solid']
tab_sign = dashstyle

name_algorithms ={
    "DSGD":"D-NAG",
    "moDSGD":"D-GD",
    "FedProxyProx":"PIGS",
}
name_attack ={
    "Optimal_InnerProductManipulation":"IPM",
    "Optimal_ALittleIsEnough":"ALIE",
    "NoAttack":"No Attack",
}

# from managers import ParamsManager

def shift_curves(ax, shift, epsilon=1e-6):
    for line in ax.lines:
        line.set_ydata(line.get_ydata() + shift + epsilon)
    return shift


def custom_dict_to_str(dictionary):
    """
    Safely convert a dictionary to a string.
    Returns an empty string if the dictionary is empty.
    """
    return '' if not dictionary else str(dictionary)


def ensure_list(value):
    """
    Ensure the given value is returned as a list.
    If it is not, wrap it in a list.
    """
    if not isinstance(value, list):
        value = [value]
    return value


def find_best_hyperparameters(path_to_results):
    """
    Find the best hyperparameters (learning rate, momentum, weight decay) 
    that maximize the minimum accuracy across different attacks.

    Reads a configuration file (config.json) in `path_to_results` 
    and writes out the best hyperparameters and the corresponding 
    step at which maximum accuracy was reached for each aggregator, each attack and each algorithm.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    path_hyperparameters = path_to_results + "/best_hyperparameters"



    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    wd_list = data["model"]["weight_decay"]

    #<-------------- Training Algorithm Config ------------->
    training_algorithms = data["benchmark_config"]["training_algorithm"]

    # <-------------- Honest Nodes Config ------------->
    # momentum_list = data["honest_clients"]["momentum"]
  

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    wd_list = ensure_list(wd_list)

    # Number of accuracy checkpoints
    nb_accuracies = 1 + math.ceil(nb_steps / evaluation_delta)

    for training_algorithm_dic in training_algorithms:
        training_algorithm = training_algorithm_dic["name"]
        training_algorithm_params = training_algorithm_dic.get("parameters", {})

        lr_list = training_algorithm_params["learning_rate"]
        momentum_list = training_algorithm_params["momentum"]

        lr_list = ensure_list(lr_list)
        momentum_list = ensure_list(momentum_list)
        
        
    
        # Main nested loops to explore configurations
        for nb_honest in nb_honest_clients:
            for nb_byzantine in nb_byz:
                
                if nb_declared[0] is None:
                    nb_declared_list = [nb_byzantine]
                else:
                    nb_declared_list = nb_declared.copy()
                    nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]

                for nb_decl in nb_declared_list:
                    if set_honest_clients_as_clients:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine

                    for data_dist in data_distributions:
                        distribution_parameter_list = ensure_list(data_dist["distribution_parameter"])
                        for distribution_parameter in distribution_parameter_list:
                            for pre_agg in pre_aggregators:
                                # Build a single name from all pre-aggregators
                                pre_agg_names_list = [p["name"] for p in pre_agg]
                                pre_agg_names = "_".join(pre_agg_names_list)

                                # Prepare arrays to store final best hyperparams & steps
                                real_hyper_parameters = np.zeros((len(aggregators), 3))
                                real_steps = np.zeros((len(aggregators), len(attacks)))

                                for k, agg in enumerate(aggregators):
                                    # We'll store max accuracy for each (lr, momentum, wd) across attacks
                                    num_combinations = len(lr_list) * len(momentum_list) * len(wd_list)
                                    max_acc_config = np.zeros((num_combinations, len(attacks)))
                                    hyper_parameters = np.zeros((num_combinations, 3))
                                    steps_max_reached = np.zeros((num_combinations, len(attacks)))

                                    index_combination = 0
                                    for lr in lr_list:
                                        for momentum in momentum_list:
                                            for wd in wd_list:
                                                # tab_acc shape: (len(attacks), nb_dd_seeds, nb_training_seeds, nb_accuracies)
                                                tab_acc = np.zeros(
                                                    (
                                                        len(attacks),
                                                        nb_data_distribution_seeds,
                                                        nb_training_seeds,
                                                        nb_accuracies
                                                    )
                                                )

                                                # Fill tab_acc with loaded accuracy files
                                                for i, attack in enumerate(attacks):
                                                    for run_dd in range(nb_data_distribution_seeds):
                                                        for run in range(nb_training_seeds):
                                                            file_name = (
                                                                f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                                f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                                f"{distribution_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                                f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                                f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                            )
                                                            acc_path = os.path.join(
                                                                path_to_results,
                                                                file_name,
                                                                f"val_accuracy_tr_seed_{run + training_seed}"
                                                                f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                            )
                                                            print(acc_path, genfromtxt(acc_path, delimiter=',').shape)
                                                            tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                                tab_acc = tab_acc.reshape(
                                                    len(attacks),
                                                    nb_data_distribution_seeds * nb_training_seeds,
                                                    nb_accuracies
                                                )
                                                
                                                # Compute average accuracy across seeds, find max
                                                for i in range(len(attacks)):
                                                    avg_accuracy = np.mean(tab_acc[i], axis=0)
                                                    idx_max = np.argmax(avg_accuracy)
                                                    max_acc_config[index_combination, i] = avg_accuracy[idx_max]
                                                    steps_max_reached[index_combination, i] = idx_max * evaluation_delta

                                                hyper_parameters[index_combination] = [lr, momentum, wd]
                                                index_combination += 1

                                    # Create path if needed
                                    if not os.path.exists(path_hyperparameters):
                                        try:
                                            os.makedirs(path_hyperparameters)
                                        except OSError as error:
                                            print(f"Error creating directory: {error}")

                                    # Find the combination that maximizes the minimum accuracy across attacks
                                    max_minimum_idx = -1
                                    max_minimum_val = -1
                                    for i in range(num_combinations):
                                        current_min = np.min(max_acc_config[i])
                                        if current_min > max_minimum_val:
                                            max_minimum_idx = i
                                            max_minimum_val = current_min

                                    real_hyper_parameters[k] = hyper_parameters[max_minimum_idx]
                                    real_steps[k] = steps_max_reached[max_minimum_idx]

                                # Save results to folder
                                hyper_parameters_folder = os.path.join(path_hyperparameters, "hyperparameters")
                                steps_folder = os.path.join(path_hyperparameters, "better_step")

                                os.makedirs(hyper_parameters_folder, exist_ok=True)
                                os.makedirs(steps_folder, exist_ok=True)

                                for i, agg in enumerate(aggregators):
                                    # Save best hyperparameters
                                    file_name_hparams = (
                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                        f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                        f"{distribution_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                    )
                                    np.savetxt(
                                        os.path.join(hyper_parameters_folder, file_name_hparams),
                                        real_hyper_parameters[i]
                                    )

                                    # Save step at which max accuracy occurs for each attack
                                    for j, attack in enumerate(attacks):
                                        file_name_steps = (
                                            f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                            f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                            f"{distribution_parameter}_{pre_agg_names}_{agg['name']}_"
                                        f"{custom_dict_to_str(attack['name'])}.txt"
                                    )
                                    step_val = np.array([real_steps[i, j]])
                                    np.savetxt(os.path.join(steps_folder, file_name_steps), step_val)




def test_accuracy_curve(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
        
        try:
            with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"ERROR reading config.json: {e}")
            return
        
        try:
            os.makedirs(path_to_plot, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory: {error}")
        
        path_to_hyperparameters = path_to_results + "/best_hyperparameters"
        

        # <-------------- Benchmark Config ------------->
        training_seed = data["benchmark_config"]["training_seed"]
        nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
        nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
        nb_byz = data["benchmark_config"]["f"]
        nb_declared = data["benchmark_config"].get("tolerated_f", None)
        data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
        data_distributions = data["benchmark_config"]["data_distribution"]
        set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
        nb_steps = data["benchmark_config"]["nb_steps"]


        # <-------------- Evaluation and Results ------------->
        evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

        # <-------------- Model Config ------------->
        model_name = data["model"]["name"]
        dataset_name = data["model"]["dataset_name"]
        

        #<-------------- Training Algorithm Config ------------->
        training_algorithms = data["benchmark_config"]["training_algorithm"]

        # <-------------- Honest Nodes Config ------------->
        # momentum_list = data["honest_clients"]["momentum"]
        wd_list = data["model"]["weight_decay"]

        # <-------------- Aggregators Config ------------->
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

        # <-------------- Attacks Config ------------->
        attacks = data["attack"]

        # Ensure certain configurations are always lists
        training_algorithms = ensure_list(training_algorithms)
        nb_honest_clients = ensure_list(nb_honest_clients)
        nb_byz = ensure_list(nb_byz)
        nb_declared = ensure_list(nb_declared)
        data_distributions = ensure_list(data_distributions)
        aggregators = ensure_list(aggregators)

        # Pre-aggregators can be multiple or single dict; unify them
        if not pre_aggregators or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]

        attacks = ensure_list(attacks)
        wd_list = ensure_list(wd_list)
        

        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
        
        for training_algorithm_dic in training_algorithms:
            training_algorithm = training_algorithm_dic["name"]
            training_algorithm_params = training_algorithm_dic.get("parameters", {})

            momentum_list = training_algorithm_params["momentum"]
            lr_list = training_algorithm_params["learning_rate"]

            lr_list = ensure_list(lr_list)
            momentum_list = ensure_list(momentum_list)
        
            for nb_honest in nb_honest_clients:
                for nb_byzantine in nb_byz:

                    if nb_declared[0] is None:
                        nb_declared_list = [nb_byzantine]
                    else:
                        nb_declared_list = nb_declared.copy()
                        nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
                    
                    for nb_decl in nb_declared_list:

                        if set_honest_clients_as_clients:
                            nb_nodes = nb_honest
                        else:
                            nb_nodes = nb_honest + nb_byzantine
                        
                        for data_dist in data_distributions:
                            dist_parameter_list = data_dist["distribution_parameter"]
                            dist_parameter_list = ensure_list(dist_parameter_list)
                            for dist_parameter in dist_parameter_list:
                                for pre_agg in pre_aggregators:
                                    pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                    pre_agg_names = "_".join(pre_agg_list_names)
                                    for agg in aggregators:

                                        hyper_file_name = (
                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                        )


                                        full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                        if os.path.exists(full_path):
                                            hyperparameters = np.loadtxt(full_path)
                                            lr = hyperparameters[0]
                                            momentum = hyperparameters[1]
                                            wd = hyperparameters[2]
                                        else:
                                            lr = lr_list[0]
                                            momentum = momentum_list[0]
                                            wd = wd_list[0]

                                        tab_acc = np.zeros((
                                            len(attacks), 
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_accuracies
                                        ))

                                        for i, attack in enumerate(attacks):
                                            for run_dd in range(nb_data_distribution_seeds):
                                                for run in range(nb_training_seeds):
                                                    file_name = (
                                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                        f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                        f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                        f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                        f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                    )
                                                    acc_path = os.path.join(
                                                        path_to_results,
                                                        file_name,
                                                        f"test_accuracy_tr_seed_{run + training_seed}"
                                                        f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                    )
                                                    tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                        tab_acc = tab_acc.reshape(
                                            len(attacks),
                                            nb_data_distribution_seeds * nb_training_seeds,
                                            nb_accuracies
                                        )
                                        
                                        err = np.zeros((len(attacks), nb_accuracies))
                                        for i in range(len(err)):
                                            err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                        
                                        plt.rcParams.update({'font.size': 12})

                                        
                                        for i, attack in enumerate(attacks):
                                            attack = attack["name"]
                                            plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = attack, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                            plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)

                                        plt.xlabel('Round')
                                        plt.ylabel('Accuracy')
                                        plt.xlim(0,(nb_accuracies-1)*evaluation_delta)
                                        plt.ylim(0,1)
                                        plt.grid()
                                        plt.legend()

                                        plot_name = (
                                            f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                            f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                            f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                        )
                                        
                                        plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                        plt.close()

def loss_heatmap(path_to_results, path_to_plot):
    """
    Creates a heatmap where the axis are the number of 
    byzantine nodes and the distribution parameter.
    Each number is the mean of the best training losses reached 
    by the model across seeds, using a specific aggregation.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters" 

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    wd_list = data["model"]["weight_decay"]
    

    #<-------------- Training Algorithm Config ------------->
    training_algorithm = data["benchmark_config"]["training_algorithm"]["name"]
    training_algorithm_params = data["benchmark_config"]["training_algorithm"].get("parameters", {})
    lr_list = training_algorithm_params["learning_rate"]
    
    momentum_list = training_algorithm_params["momentum"]

    # <-------------- Honest Nodes Config ------------->
    

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False

    for pre_agg in pre_aggregators:

        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)

        for agg in aggregators:

            for data_dist in data_distributions:

                distribution_parameter_list = data_dist["distribution_parameter"]
                distribution_parameter_list = ensure_list(distribution_parameter_list)

                for nb_honest in nb_honest_clients:

                    for nb_decl in nb_declared:
                        actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):

                                hyper_file_name = (
                                    f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_{pre_agg_names}_{agg['name']}.txt"
                                )

                                
                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                lowest_loss = 0
                                for attack in attacks:

                                    config_file_name = (
                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                        f"{custom_dict_to_str(attack['name'])}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )

                                    try:
                                        with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                            data = json.load(file)
                                    except Exception as e:
                                        print("ERROR: "+ str(e))

                                    nb_steps = data["benchmark_config"]["nb_steps"]

                                    losses = np.zeros(
                                        (
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_steps
                                        )
                                    )

                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            losses[run_dd][run] = genfromtxt(
                                                f"{path_to_results}/{config_file_name}/"
                                                f"train_loss_tr_seed_{run + training_seed}_"
                                                f"dd_seed_{run_dd + data_distribution_seed}.txt",
                                                delimiter=','
                                            )

                                    losses = losses.reshape(
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_steps
                                    )

                                    losses = np.mean(losses, axis=0)

                                    temp_lowest_loss = np.min(losses)

                                    if temp_lowest_loss > lowest_loss:
                                        lowest_loss = temp_lowest_loss
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = lowest_loss

                        if declared_equal_real:
                            end_file_name = "tolerated_f_equal_real.pdf"
                        else:
                            end_file_name = f"tolerated_f_{nb_decl}.pdf"

                        file_name = (
                            f"train_loss_{dataset_name}_"
                            f"_{training_algorithm}_"
                            f"{model_name}_"
                            f"{custom_dict_to_str(data_dist['name'])}_"
                            f"{pre_agg_names}_"
                            f"{agg['name']}_"
                            f"nb_honest_clients_{nb_honest}_"
                            + end_file_name
                        )

                    
                        column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                        row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                        column_names.reverse()

                        try:
                            os.makedirs(path_to_plot, exist_ok=True)
                        except OSError as error:
                            print(f"Error creating directory: {error}")

                        sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, cmap=sns.cm.rocket_r, annot=True)
                        plt.xlabel("Number of Byzantine clients")
                        plt.ylabel("Data heterogeneity level")
                        plt.tight_layout()
                        plt.savefig(path_to_plot +"/"+ file_name)
                        plt.close()


def test_heatmap(path_to_results, path_to_plot):
    """
    Creates a heatmap where the axis are the number of 
    byzantine nodes and the distribution parameter.
    Each number is the mean of the best accuracy reached 
    by the model across seeds, using a specific aggregation.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    

    #<-------------- Training Algorithm Config ------------->
    training_algorithm = data["benchmark_config"]["training_algorithm"]["name"]
    training_algorithm_params = data["benchmark_config"]["training_algorithm"].get("parameters", {})
    lr_list = training_algorithm_params["learning_rate"]    
    wd_list = data["model"]["weight_decay"]
    momentum_list = training_algorithm_params["momentum"]

    # <-------------- Honest Nodes Config ------------->
    # momentum_list = data["honest_clients"]["momentum"]
    

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False

    for pre_agg in pre_aggregators:

        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)

        for agg in aggregators:

            for data_dist in data_distributions:

                distribution_parameter_list = data_dist["distribution_parameter"]
                distribution_parameter_list = ensure_list(distribution_parameter_list)

                for nb_honest in nb_honest_clients:

                    for nb_decl in nb_declared:
                        actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):

                                hyper_file_name = (
                                    f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_{pre_agg_names}_{agg['name']}.txt"
                                )

                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                worst_accuracy = np.inf
                                for attack in attacks:

                                    config_file_name = (
                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                        f"{custom_dict_to_str(attack['name'])}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )

                                    try:
                                        with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                            data = json.load(file)
                                    except Exception as e:
                                        print("ERROR: "+ str(e))

                                    nb_steps = data["benchmark_config"]["nb_steps"]
                                    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

                                    tab_acc = np.zeros(
                                        (
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_accuracies
                                        )
                                    )

                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            tab_acc[run_dd][run] = genfromtxt(
                                                f"{path_to_results}/{config_file_name}/"
                                                f"test_accuracy_tr_seed_{run + training_seed}_"
                                                f"dd_seed_{run_dd + data_distribution_seed}.txt",
                                                delimiter=','
                                            )

                                    tab_acc = tab_acc.reshape(
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )

                                    tab_acc = tab_acc.mean(axis=0)

                                    accuracy = np.max(tab_acc)

                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy
                    
                        if declared_equal_real:
                            end_file_name = "tolerated_f_equal_real.pdf"
                        else:
                            end_file_name = f"tolerated_f_{nb_decl}.pdf"

                        file_name = (
                            f"test_{dataset_name}_"
                            f"{model_name}_{training_algorithm}_"
                            f"{custom_dict_to_str(data_dist['name'])}_"
                            f"{pre_agg_names}_"
                            f"{agg['name']}_"
                            f"nb_honest_clients_{nb_honest}_"
                            + end_file_name
                        )

                    
                        column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                        row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                        column_names.reverse()

                        try:
                            os.makedirs(path_to_plot, exist_ok=True)
                        except OSError as error:
                            print(f"Error creating directory: {error}")

                        sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                        plt.xlabel("Number of Byzantine clients")
                        plt.ylabel("Data heterogeneity level")
                        plt.tight_layout()
                        plt.savefig(path_to_plot +"/"+ file_name)
                        plt.close()


def aggregated_test_heatmap(path_to_results, path_to_plot):
    """
    Heatmap with the aggregated info of all aggregators, 
    for every region in the heatmap, it shows the aggregation 
    with the best accuracy.
    """
    try:
        with open(path_to_results+'/config.json', 'r') as file:
            data = json.load(file)
    except Exception as e:
        print("ERROR: "+ str(e))

    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    path_to_hyperparameters = path_to_results + "/best_hyperparameters"
    
    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    wd_list = data["model"]["weight_decay"]

    #<-------------- Training Algorithm Config ------------->
    training_algorithm = data["benchmark_config"]["training_algorithm"]["name"]
    training_algorithm_params = data["benchmark_config"]["training_algorithm"].get("parameters", {})

    lr_list = training_algorithm_params.get("learning_rate", [])
    momentum_list = training_algorithm_params.get("momentum", [])

    # <-------------- Honest Nodes Config ------------->
    

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    if nb_declared[0] is None:
        declared_equal_real = True
        nb_declared = [nb_byz[-1]]
    else:
        declared_equal_real = False
    
    for pre_agg in pre_aggregators:

        for nb_honest in nb_honest_clients:

            for nb_decl in nb_declared:
                actual_nb_byz = [item for item in nb_byz if item <= nb_decl]
                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                pre_agg_names = "_".join(pre_agg_list_names)

                for data_dist in data_distributions:

                    data_dist["distribution_parameter"] = ensure_list(data_dist["distribution_parameter"])
                    distribution_parameter_list = data_dist["distribution_parameter"]
                    heat_map_cube = np.zeros((len(aggregators), len(distribution_parameter_list), len(actual_nb_byz)))

                    for z, agg in enumerate(aggregators):

                        heat_map_table = np.zeros((len(distribution_parameter_list), len(actual_nb_byz)))

                        for y, nb_byzantine in enumerate(actual_nb_byz):

                            if declared_equal_real:
                                nb_decl = nb_byzantine

                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                                nb_honest = nb_nodes - nb_byzantine
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):

                                hyper_file_name = (
                                    f"{dataset_name}_"
                                    f"{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                    f"{pre_agg_names}_{agg['name']}.txt"
                                )

                                
                                full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                if os.path.exists(full_path):
                                    hyperparameters = np.loadtxt(full_path)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    wd = hyperparameters[2]
                                else:
                                    lr = lr_list[0]
                                    momentum = momentum_list[0]
                                    wd = wd_list[0]

                                
                                worst_accuracy = np.inf
                                for attack in attacks:
                                    config_file_name = (
                                        f"{dataset_name}_"
                                        f"{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                        f"{custom_dict_to_str(attack['name'])}_"
                                        f"lr_{lr}_"
                                        f"mom_{momentum}_"
                                        f"wd_{wd}"
                                    )

                                    try:
                                        with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                            data = json.load(file)
                                    except Exception as e:
                                        print("ERROR: "+ str(e))

                                    nb_steps = data["benchmark_config"]["nb_steps"]
                                    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

                                    tab_acc = np.zeros(
                                        (
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_accuracies
                                        )
                                    )

                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            tab_acc[run_dd][run] = genfromtxt(
                                                f"{path_to_results}/{config_file_name}/"
                                                f"test_accuracy_tr_seed_{run + training_seed}_"
                                                f"dd_seed_{run_dd + data_distribution_seed}.txt",
                                                delimiter=','
                                            )

                                    tab_acc = tab_acc.reshape(
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )
                                    
                                    tab_acc = tab_acc.mean(axis=0)
                                    accuracy = np.max(tab_acc)

                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

                        heat_map_cube[z] = heat_map_table
                    

                    if declared_equal_real:
                        end_file_name = "tolerated_f_equal_real.pdf"
                    else:
                        end_file_name = f"tolerated_f_{nb_decl}.pdf"

                    file_name = (
                        f"best_test_{dataset_name}_"
                        f"{model_name}_{training_algorithm}_"
                        f"{custom_dict_to_str(data_dist['name'])}_"
                        f"{pre_agg_names}_"
                        f"nb_honest_clients_{nb_honest}_"
                        + end_file_name
                    )
                    
                    column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                    row_names = [str(nb_byzantine) for nb_byzantine in actual_nb_byz]
                    column_names.reverse()

                    heat_map_table = np.max(heat_map_cube, axis=0)
                    sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                    plt.xlabel("Number of Byzantine clients")
                    plt.ylabel("Data heterogeneity level")
                    plt.tight_layout()
                    plt.savefig(path_to_plot +"/"+ file_name)
                    plt.close()



#### Comparison of different optimization algorithms as well


def train_loss_curve_alg(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
        
        try:
            with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"ERROR reading config.json: {e}")
            return
        
        try:
            os.makedirs(path_to_plot, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory: {error}")
        
        path_to_hyperparameters = path_to_results + "/best_hyperparameters"
        

        # <-------------- Benchmark Config ------------->
        training_seed = data["benchmark_config"]["training_seed"]
        nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
        nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
        nb_byz = data["benchmark_config"]["f"]
        nb_declared = data["benchmark_config"].get("tolerated_f", None)
        data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
        data_distributions = data["benchmark_config"]["data_distribution"]
        set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
        nb_steps = data["benchmark_config"]["nb_steps"]


        # <-------------- Evaluation and Results ------------->
        evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

        # <-------------- Model Config ------------->
        model_name = data["model"]["name"]
        dataset_name = data["model"]["dataset_name"]
        wd_list = data["model"]["weight_decay"]

        #<-------------- Training Algorithm Config ------------->
        training_algorithms = data["benchmark_config"]["training_algorithm"]


        # <-------------- Honest Nodes Config ------------->
        

        # <-------------- Aggregators Config ------------->
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

        # <-------------- Attacks Config ------------->
        attacks = data["attack"]

        # Ensure certain configurations are always lists
        training_algorithms = ensure_list(training_algorithms)
        nb_honest_clients = ensure_list(nb_honest_clients)
        nb_byz = ensure_list(nb_byz)
        nb_declared = ensure_list(nb_declared)
        data_distributions = ensure_list(data_distributions)
        aggregators = ensure_list(aggregators)

        # Pre-aggregators can be multiple or single dict; unify them
        if not pre_aggregators or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]

        attacks = ensure_list(attacks)
        wd_list = ensure_list(wd_list)

        nb_steps = data["benchmark_config"]["nb_steps"]

        for nb_honest in nb_honest_clients:
            for nb_byzantine in nb_byz:
                
                if nb_declared[0] is None:
                    nb_declared_list = [nb_byzantine]
                else:
                    nb_declared_list = nb_declared.copy()
                    nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]

                for nb_decl in nb_declared_list:
                    if set_honest_clients_as_clients:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine

                    for data_dist in data_distributions:
                        dist_parameter_list = data_dist["distribution_parameter"]
                        dist_parameter_list = ensure_list(dist_parameter_list)
                        for dist_parameter in dist_parameter_list:
                            for pre_agg in pre_aggregators:
                                # Build a single name from all pre-aggregators
                                pre_agg_list_names = [p["name"] for p in pre_agg]
                                pre_agg_names = "_".join(pre_agg_list_names)

                                for agg in aggregators:
                                    # Choose the first attack for plotting
                                    attack = attacks[0]

                                    tab_loss = {}
                                    errs = {}

                                    for training_algorithm_dic in training_algorithms:
                                        training_algorithm = training_algorithm_dic["name"]
                                        training_algorithm_params = training_algorithm_dic.get("parameters", {})

                                        lr_list = training_algorithm_params["learning_rate"]
                                        
                                        momentum_list = training_algorithm_params["momentum"]

                                        lr_list = ensure_list(lr_list)
                                        momentum_list = ensure_list(momentum_list)
            
                                        
                                        hyper_file_name = (
                                            f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                            f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                        )

                                        full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                        if os.path.exists(full_path):
                                            hyperparameters = np.loadtxt(full_path)
                                            lr = hyperparameters[0]
                                            momentum = hyperparameters[1]
                                            wd = hyperparameters[2]
                                        else:
                                            lr = lr_list[0]
                                            momentum = momentum_list[0]
                                            wd = wd_list[0]

                                        config_file_name = (
                                            f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                            f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                            f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                            f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                            f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                        )

                                        losses = np.zeros((
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_steps
                                        ))

                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                loss_file = os.path.join(
                                                    path_to_results,
                                                    config_file_name,
                                                    f"train_loss_tr_seed_{run + training_seed}_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                )
                                                losses[run_dd, run] = genfromtxt(loss_file, delimiter=',')

                                        losses = losses.reshape(
                                            nb_data_distribution_seeds * nb_training_seeds,
                                            nb_steps
                                        )

                                        mean_loss = np.mean(losses, axis=0)
                                        tab_loss[training_algorithm] = mean_loss

                                        err = (1.96 * np.std(losses, axis=0)) / math.sqrt(nb_training_seeds * nb_data_distribution_seeds)
                                        errs[training_algorithm] = err

                                    plt.rcParams.update({'font.size': 12})

                                    for i, training_algorithm_dic in enumerate(training_algorithms):
                                        training_algorithm = training_algorithm_dic["name"]
                                        plt.plot(
                                            np.arange(nb_steps), 
                                            tab_loss[training_algorithm], 
                                            label=training_algorithm, 
                                            color=colors[i % len(colors)], 
                                            linestyle=tab_sign[i % len(tab_sign)], 
                                            marker=markers[i % len(markers)], 
                                            markevery=1
                                        )
                                        plt.fill_between(
                                            np.arange(nb_steps), 
                                            tab_loss[training_algorithm] - errs[training_algorithm], 
                                            tab_loss[training_algorithm] + errs[training_algorithm], 
                                            alpha=0.25
                                        )

                                    plt.xlabel('Step')
                                    plt.ylabel('Train Loss')
                                    plt.xlim(0, nb_steps - 1)
                                    plt.grid()
                                    plt.legend()

                                    plot_name = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_{custom_dict_to_str(attack['name'])}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )

                                    plt.savefig(path_to_plot + "/" + plot_name + '_train_loss.pdf')
                                    plt.close()


def test_accuracy_curve_alg(path_to_results, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
        
        try:
            with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"ERROR reading config.json: {e}")
            return
        
        try:
            os.makedirs(path_to_plot, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory: {error}")
        
        path_to_hyperparameters = path_to_results + "/best_hyperparameters"
        

        # <-------------- Benchmark Config ------------->
        training_seed = data["benchmark_config"]["training_seed"]
        nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
        nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
        nb_byz = data["benchmark_config"]["f"]
        nb_declared = data["benchmark_config"].get("tolerated_f", None)
        data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
        data_distributions = data["benchmark_config"]["data_distribution"]
        set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
        nb_steps = data["benchmark_config"]["nb_steps"]


        # <-------------- Evaluation and Results ------------->
        evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

        # <-------------- Model Config ------------->
        model_name = data["model"]["name"]
        dataset_name = data["model"]["dataset_name"]
        wd_list = data["model"]["weight_decay"]

        #<-------------- Training Algorithm Config ------------->
        training_algorithms = data["benchmark_config"]["training_algorithm"]

        # <-------------- Honest Nodes Config ------------->
        

        # <-------------- Aggregators Config ------------->
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

        # <-------------- Attacks Config ------------->
        attacks = data["attack"]

        # Ensure certain configurations are always lists
        training_algorithms = ensure_list(training_algorithms)
        nb_honest_clients = ensure_list(nb_honest_clients)
        nb_byz = ensure_list(nb_byz)
        nb_declared = ensure_list(nb_declared)
        data_distributions = ensure_list(data_distributions)
        aggregators = ensure_list(aggregators)
        wd_list = ensure_list(wd_list)

        # Pre-aggregators can be multiple or single dict; unify them
        if not pre_aggregators or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]

        attacks = ensure_list(attacks)

        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
        
        for nb_honest in nb_honest_clients:
            for nb_byzantine in nb_byz:

                if nb_declared[0] is None:
                    nb_declared_list = [nb_byzantine]
                else:
                    nb_declared_list = nb_declared.copy()
                    nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
                
                for nb_decl in nb_declared_list:

                    if set_honest_clients_as_clients:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine
                    
                    for data_dist in data_distributions:
                        dist_parameter_list = data_dist["distribution_parameter"]
                        dist_parameter_list = ensure_list(dist_parameter_list)
                        for dist_parameter in dist_parameter_list:
                            for pre_agg in pre_aggregators:
                                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                pre_agg_names = "_".join(pre_agg_list_names)
                                for agg in aggregators:

                                    # Choose the first attack for plotting
                                    attack = attacks[0]

                                    tab_acc = {}
                                    errs = {}

                                    for training_algorithm_dic in training_algorithms:
                                        training_algorithm = training_algorithm_dic["name"]
                                        training_algorithm_params = training_algorithm_dic.get("parameters", {})

                                        lr_list = training_algorithm_params["learning_rate"]
                                        momentum_list = training_algorithm_params["momentum"]

                                        lr_list = ensure_list(lr_list)
                                        momentum_list = ensure_list(momentum_list)


                                        hyper_file_name = (
                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                        )


                                        full_path = os.path.join(path_to_hyperparameters, "hyperparameters", hyper_file_name)

                                        if os.path.exists(full_path):
                                            hyperparameters = np.loadtxt(full_path)
                                            lr = hyperparameters[0]
                                            momentum = hyperparameters[1]
                                            wd = hyperparameters[2]
                                        else:
                                            lr = lr_list[0]
                                            momentum = momentum_list[0]
                                            wd = wd_list[0]

                                        tab_acc_temp = np.zeros((
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_accuracies
                                        ))

                                        for i in range(nb_data_distribution_seeds):
                                            for j in range(nb_training_seeds):
                                                file_name = (
                                                    f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                    f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                    f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                    f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                )
                                                acc_path = os.path.join(
                                                    path_to_results,
                                                    file_name,
                                                    f"test_accuracy_tr_seed_{j + training_seed}"
                                                    f"_dd_seed_{i + data_distribution_seed}.txt"
                                                )
                                                tab_acc_temp[i, j] = genfromtxt(acc_path, delimiter=',')

                                        tab_acc_temp = tab_acc_temp.reshape(
                                            nb_data_distribution_seeds * nb_training_seeds,
                                            nb_accuracies
                                        )
                                        
                                        mean_acc = np.mean(tab_acc_temp, axis=0)
                                        tab_acc[training_algorithm] = mean_acc
                                        
                                        err = (1.96*np.std(tab_acc_temp, axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                        errs[training_algorithm] = err

                                    plt.rcParams.update({'font.size': 12})

                                    
                                    for i, training_algorithm_dic in enumerate(training_algorithms):
                                        training_algorithm = training_algorithm_dic["name"]
                                        plt.plot(np.arange(nb_accuracies)*evaluation_delta, tab_acc[training_algorithm], label = training_algorithm, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                        plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, tab_acc[training_algorithm] - errs[training_algorithm], tab_acc[training_algorithm] + errs[training_algorithm], alpha = 0.25)

                                    plt.xlabel('Round')
                                    plt.ylabel('Accuracy')
                                    plt.xlim(0,(nb_accuracies-1)*evaluation_delta)
                                    plt.ylim(0,1)
                                    plt.grid()
                                    plt.legend()

                                    plot_name = (
                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_{custom_dict_to_str(attack['name'])}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                    )
                                    
                                    plt.savefig(path_to_plot+"/"+plot_name+'_test_accuracy.pdf')
                                    plt.close()


def test_accuracy_curve_hyperparameter(path_to_results, path_to_plot):
    """
    Test accuracy curve for different hyperparameters of a given training algorithm.
    """
    
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    wd_list = data["model"]["weight_decay"]

    #<-------------- Training Algorithm Config ------------->
    training_algorithms = data["benchmark_config"]["training_algorithm"]

    # <-------------- Honest Nodes Config ------------->
    

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    training_algorithms = ensure_list(training_algorithms)
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)
    wd_list = ensure_list(wd_list)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)

    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
    
    for training_algorithm_dic in training_algorithms:
        training_algorithm = training_algorithm_dic["name"]
        training_algorithm_params = training_algorithm_dic.get("parameters", {})

        lr_list = training_algorithm_params["learning_rate"]
        momentum_list = training_algorithm_params["momentum"]

        lr_list = ensure_list(lr_list)
        momentum_list = ensure_list(momentum_list)
    
        for nb_honest in nb_honest_clients:
            for nb_byzantine in nb_byz:

                if nb_declared[0] is None:
                    nb_declared_list = [nb_byzantine]
                else:
                    nb_declared_list = nb_declared.copy()
                    nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
                
                for nb_decl in nb_declared_list:

                    if set_honest_clients_as_clients:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine
                    
                    for data_dist in data_distributions:
                        dist_parameter_list = data_dist["distribution_parameter"]
                        dist_parameter_list = ensure_list(dist_parameter_list)
                        for dist_parameter in dist_parameter_list:
                            for pre_agg in pre_aggregators:
                                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                pre_agg_names = "_".join(pre_agg_list_names)
                                for agg in aggregators:

                                    # Choose the first attack for plotting
                                    attack = attacks[0]

                                    

                                    tab_acc = np.zeros((
                                        len(lr_list), 
                                        len(momentum_list),
                                        len(wd_list),
                                        nb_data_distribution_seeds,
                                        nb_training_seeds,
                                        nb_accuracies
                                    ))
                                    for i_lr, lr in enumerate(lr_list):
                                        for i_mom, momentum in enumerate(momentum_list):
                                            for i_wd, wd in enumerate(wd_list):
                                                i = i_lr * len(momentum_list) * len(wd_list) + i_mom * len(wd_list) + i_wd
                                            
                                                for run_dd in range(nb_data_distribution_seeds):
                                                    for run in range(nb_training_seeds):
                                                        file_name = (
                                                            f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                            f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                            f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                            f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                            f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                        )
                                                        acc_path = os.path.join(
                                                            path_to_results,
                                                            file_name,
                                                            f"test_accuracy_tr_seed_{run + training_seed}"
                                                            f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                        )
                                                        tab_acc[i_lr, i_mom, i_wd, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                    tab_acc = tab_acc.reshape(
                                        len(lr_list),
                                        len(momentum_list),
                                        len(wd_list),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )

                                    err = np.zeros((len(lr_list), len(momentum_list), len(wd_list), nb_accuracies))
                                    for i_lr in range(len(lr_list)):
                                        for i_mom in range(len(momentum_list)):
                                            for i_wd in range(len(wd_list)):
                                                err[i_lr, i_mom, i_wd] = (1.96*np.std(tab_acc[i_lr, i_mom, i_wd], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)

                                    plt.rcParams.update({'font.size': 12})


                                    for i_lr, lr in enumerate(lr_list):
                                        for i_mom, momentum in enumerate(momentum_list):
                                            for i_wd, wd in enumerate(wd_list):
                                                i = i_lr * len(momentum_list) * len(wd_list) + i_mom * len(wd_list) + i_wd
                                                plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i_lr, i_mom, i_wd], axis = 0), label = f"lr={lr}, mom={momentum}, wd={wd}", color = colors[i%len(colors)], linestyle = tab_sign[i%len(tab_sign)], marker = markers[i%len(markers)], markevery = 1)
                                                plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i_lr, i_mom, i_wd], axis = 0) - err[i_lr, i_mom, i_wd], np.mean(tab_acc[i_lr, i_mom, i_wd], axis = 0) + err[i_lr, i_mom, i_wd], alpha = 0.25)

                                    plt.xlabel('Round')
                                    plt.ylabel('Accuracy')
                                    plt.xlim(0,(nb_accuracies-1)*evaluation_delta)
                                    plt.ylim(0,1)
                                    plt.grid()
                                    plt.legend()

                                    plot_name = (
                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_{custom_dict_to_str(attack['name'])}"
                                    )
                                    
                                    plt.savefig(path_to_plot+"/"+plot_name+'_hyper_test_accuracy.pdf')
                                    plt.close()



def train_loss_curve_hyperparameter(path_to_results, path_to_plot):
    """
    Test accuracy curve for different hyperparameters of a given training algorithm.
    """
    
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]


    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    wd_list = data["model"]["weight_decay"]

    #<-------------- Training Algorithm Config ------------->
    training_algorithms = data["benchmark_config"]["training_algorithm"]

    # <-------------- Honest Nodes Config ------------->
    

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    training_algorithms = ensure_list(training_algorithms)
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)
    wd_list = ensure_list(wd_list)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)

    nb_steps = data["benchmark_config"]["nb_steps"]
    
    for training_algorithm_dic in training_algorithms:
        training_algorithm = training_algorithm_dic["name"]
        training_algorithm_params = training_algorithm_dic.get("parameters", {})

        lr_list = training_algorithm_dic["parameters"]["learning_rate"]
        momentum_list = training_algorithm_dic["parameters"]["momentum"]

        lr_list = ensure_list(lr_list)
        momentum_list = ensure_list(momentum_list)
        
    
        for nb_honest in nb_honest_clients:
            for nb_byzantine in nb_byz:

                if nb_declared[0] is None:
                    nb_declared_list = [nb_byzantine]
                else:
                    nb_declared_list = nb_declared.copy()
                    nb_declared_list = [item for item in nb_declared_list if item >= nb_byzantine]
                
                for nb_decl in nb_declared_list:

                    if set_honest_clients_as_clients:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine
                    
                    for data_dist in data_distributions:
                        dist_parameter_list = data_dist["distribution_parameter"]
                        dist_parameter_list = ensure_list(dist_parameter_list)
                        for dist_parameter in dist_parameter_list:
                            for pre_agg in pre_aggregators:
                                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                pre_agg_names = "_".join(pre_agg_list_names)
                                for agg in aggregators:

                                    # Choose the first attack for plotting
                                    attack = attacks[0]

                                    

                                    tab_acc = np.zeros((
                                        len(lr_list), 
                                        len(momentum_list),
                                        len(wd_list),
                                        nb_data_distribution_seeds,
                                        nb_training_seeds,
                                        nb_steps
                                    ))
                                    for i_lr, lr in enumerate(lr_list):
                                        for i_mom, momentum in enumerate(momentum_list):
                                            for i_wd, wd in enumerate(wd_list):
                                                i = i_lr * len(momentum_list) * len(wd_list) + i_mom * len(wd_list) + i_wd
                                            
                                                for run_dd in range(nb_data_distribution_seeds):
                                                    for run in range(nb_training_seeds):
                                                        file_name = (
                                                            f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                            f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                            f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                            f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                            f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                        )
                                                        acc_path = os.path.join(
                                                            path_to_results,
                                                            file_name,
                                                            f"train_loss_tr_seed_{run + training_seed}"
                                                            f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                        )
                                                        tab_acc[i_lr, i_mom, i_wd, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                    tab_acc = tab_acc.reshape(
                                        len(lr_list),
                                        len(momentum_list),
                                        len(wd_list),
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_steps
                                    )

                                    err = np.zeros((len(lr_list), len(momentum_list), len(wd_list), nb_steps))
                                    for i_lr in range(len(lr_list)):
                                        for i_mom in range(len(momentum_list)):
                                            for i_wd in range(len(wd_list)):
                                                err[i_lr, i_mom, i_wd] = (1.96*np.std(tab_acc[i_lr, i_mom, i_wd], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)

                                    plt.rcParams.update({'font.size': 12})


                                    for i_lr, lr in enumerate(lr_list):
                                        for i_mom, momentum in enumerate(momentum_list):
                                            for i_wd, wd in enumerate(wd_list):
                                                i = i_lr * len(momentum_list) * len(wd_list) + i_mom * len(wd_list) + i_wd
                                                plt.plot(np.arange(nb_steps)*evaluation_delta, np.mean(tab_acc[i_lr, i_mom, i_wd], axis = 0), label = f"lr={lr}, mom={momentum}, wd={wd}", color = colors[i%len(colors)], linestyle = tab_sign[i%len(tab_sign)], marker = markers[i%len(markers)], markevery = 1)
                                                plt.fill_between(np.arange(nb_steps)*evaluation_delta, np.mean(tab_acc[i_lr, i_mom, i_wd], axis = 0) - err[i_lr, i_mom, i_wd], np.mean(tab_acc[i_lr, i_mom, i_wd], axis = 0) + err[i_lr, i_mom, i_wd], alpha = 0.25)
                                    plt.xlabel('Step')
                                    plt.ylabel('Train Loss')
                                    plt.xlim(0, nb_steps - 1)
                                    plt.ylim(0,np.mean(tab_acc[i_lr, i_mom, i_wd], axis = 0)[0]*1.1)
                                    plt.grid()
                                    plt.legend()

                                    plot_name = (
                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_decl}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_{custom_dict_to_str(attack['name'])}"
                                    )
                                    
                                    plt.savefig(path_to_plot+"/"+plot_name+'_hyper_train_loss.pdf')
                                    plt.close()

def training_algorithms_comparison_curve(path_to_results, path_to_plot, metric='test_accuracy', use_ref=False, path_to_results_ref = None):
    """
    Plot comparison of different training algorithms with their different hyperparameters on the same plot for the specified metric.
    Similar to train_loss_curve_hyperparameter but for multiple algorithms.
    """
    
    if metric not in ['test_accuracy', 'train_accuracy', 'train_loss']:
        raise ValueError("metric must be 'test_accuracy', 'train_accuracy', or 'train_loss'")
    
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")

    if use_ref:
        if path_to_results_ref is None:
            path_to_results_ref = path_to_results
        try:
            with open(os.path.join(path_to_results_ref, 'ref_config.json'), 'r') as file:
                ref_config = json.load(file)
        except Exception as e:
            print(f"ERROR reading ref_config.json: {e}")
            return

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]
    
    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]
    
    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    wd_list = data["model"]["weight_decay"]
    
    #<-------------- Training Algorithm Config ------------->
    training_algorithms = data["benchmark_config"]["training_algorithm"]
    
    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]
    
    # <-------------- Attacks Config ------------->
    attacks = data["attack"]
    
    # Ensure certain configurations are always lists
    training_algorithms = ensure_list(training_algorithms)
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)
    
    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]
    
    attacks = ensure_list(attacks)
    wd_list = ensure_list(wd_list)
    
    
    
    # Assume single values for simplicity, for the following parameters
    nb_honest = nb_honest_clients[0]

    pre_agg = pre_aggregators[0]
    pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
    pre_agg_names = "_".join(pre_agg_list_names)

    for nb_byzantine in nb_byz:
        for wd in wd_list:
            for attack in attacks:
                for data_dist in data_distributions:
                    for dist_parameter in ensure_list(data_dist["distribution_parameter"]):
                        
                        if metric == 'test_accuracy' or metric == 'train_accuracy':
                            nb_points = 1 + math.ceil(nb_steps / evaluation_delta)
                            xlabel = 'Round'
                            ylabel = 'Accuracy'
                            ylim_upper = 0.94
                            ylim_lower = 0.9
                            file_key = 'test_accuracy_tr_seed' if metric == 'test_accuracy' else 'val_accuracy_tr_seed'
                        else:
                            nb_points = nb_steps
                            xlabel = 'Step'
                            ylabel = 'Train Loss'
                            ylim_upper = -np.inf
                            ylim_lower = np.inf
                            file_key = 'train_loss_tr_seed'

                        ### Selection of the reference plot

                        if use_ref:
                            training_algorithms_ref = ensure_list(ref_config["benchmark_config"]['training_algorithm'])
                            nb_honest_ref = ensure_list(ref_config['benchmark_config']['nb_honest_clients'])[0]
                            nb_byzantine_ref = 0
                            nb_decl_ref = 0
                            for training_algorithm_dic in training_algorithms_ref:

                                training_algorithm_name = training_algorithm_dic['name']
                                lrs_ref = ensure_list(training_algorithm_dic['parameters']['learning_rate'])
                                for lr_ref in lrs_ref:
                                    momentum_ref = ensure_list(training_algorithm_dic['parameters'].get('momentum', 0.0))[0]
                                    
                                    file_ref_name = (
                                            f"{dataset_name}_{model_name}_{training_algorithm_name}_n_{nb_honest_ref}_f_{nb_byzantine_ref}_"
                                            f"d_{nb_decl_ref}_{custom_dict_to_str(data_dist['name'])}_"
                                            f"{dist_parameter}_Average_"
                                            f"_NoAttack_"
                                            f"lr_{lr_ref}_mom_{momentum_ref}_wd_{wd}"
                                        )
                                    run, run_dd = 0,0
                                    data_path_ref = os.path.join(
                                            path_to_results,
                                            file_ref_name,
                                            f"{file_key}_{run + training_seed}"
                                            f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                        )
                                    tab_data_ref = genfromtxt(data_path_ref, delimiter=',')
                                    if metric in ['train_loss']:
                                        ylim_lower = np.min(tab_data_ref)

                        ### 

                        for agg in aggregators:

                            nb_decl = nb_declared[0] if nb_declared[0] is not None else nb_byzantine
                            
                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                            else:
                                nb_nodes = nb_honest + nb_byzantine
                            
                            # First pass to determine y-limits
                            for training_algorithm_dic in training_algorithms:
                                training_algorithm = training_algorithm_dic["name"]
                                training_algorithm_params = training_algorithm_dic.get("parameters", {})
                                
                                lr_list = ensure_list(training_algorithm_params.get("learning_rate", [0.01]))
                                momentum_list = ensure_list(training_algorithm_params.get("momentum", [0.0]))
                                

                                for lr in lr_list:
                                    for momentum in momentum_list:
                                            tab_data = np.zeros((
                                                nb_data_distribution_seeds,
                                                nb_training_seeds,
                                                nb_points
                                            ))
                                            
                                            for run_dd in range(nb_data_distribution_seeds):
                                                for run in range(nb_training_seeds):
                                                    file_name = (
                                                        f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                        f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                        f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                        f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                        f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                    )
                                                    data_path = os.path.join(
                                                        path_to_results,
                                                        file_name,
                                                        f"{file_key}_{run + training_seed}"
                                                        f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                    )
                                                    tab_data[run_dd, run] = genfromtxt(data_path, delimiter=',')
                                            
                                            tab_data = tab_data.reshape(
                                                nb_data_distribution_seeds * nb_training_seeds,
                                                nb_points
                                            )
                                                                                    
                                            mean_data = np.mean(tab_data, axis = 0)
                                            
                                            if metric == "train_loss":
                                                ylim_lower = np.min([np.min(mean_data), ylim_lower])      
                                                ylim_upper = np.max([mean_data[0]*1.1, ylim_upper])
                            plt.figure()
                            
                            color_index = 0
                            for training_algorithm_dic in training_algorithms:
                                training_algorithm = training_algorithm_dic["name"]
                                training_algorithm_params = training_algorithm_dic.get("parameters", {})
                                
                                lr_list = ensure_list(training_algorithm_params.get("learning_rate", [0.01]))
                                momentum_list = ensure_list(training_algorithm_params.get("momentum", [0.0]))
                                
                                for lr in lr_list:
                                    for momentum in momentum_list:
                                        
                                        tab_data = np.zeros((
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_points
                                        ))
                                        
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                file_name = (
                                                    f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                    f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                    f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                    f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                )
                                                data_path = os.path.join(
                                                    path_to_results,
                                                    file_name,
                                                    f"{file_key}_{run + training_seed}"
                                                    f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                )
                                                tab_data[run_dd, run] = genfromtxt(data_path, delimiter=',')
                                        
                                        tab_data = tab_data.reshape(
                                            nb_data_distribution_seeds * nb_training_seeds,
                                            nb_points
                                        )
                                        
                                        err = (1.96*np.std(tab_data, axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                        
                                        mean_data = np.mean(tab_data, axis = 0)

                                        if metric in ['train_loss']:
                                            mean_data = mean_data - ylim_lower
                                        else:
                                            mean_data = mean_data
                                        
                                        label = f"{name_algorithms[training_algorithm]}: lr={lr}"
                                        if momentum > 0.0:
                                            label += f", ß={momentum}"
                                        
                                        plt.plot(np.arange(nb_points)* (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1), 
                                                mean_data, label=label, color=colors[color_index % len(colors)], linestyle=tab_sign[color_index % len(tab_sign)], marker=markers[color_index % len(markers)], alpha = 0.8, markevery=1)
                                        plt.fill_between(np.arange(nb_points)* (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1), 
                                                        mean_data - err, mean_data + err, alpha=0.25)
                                        color_index += 1

                            
                                                
                            
                            plt.xlabel(xlabel)
                            plt.ylabel(ylabel)
                            plt.xlim(0, (nb_points-1) * (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1))
                            
                            
                            # if use_ref and metric not in ['train_loss']:
                            #         plt.hlines(np.max(tab_data_ref), 0,(nb_points-1) * (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1),
                            #                 colors='red', linestyles='dashed', label='Reference Loss')
                        
                            if metric in ['train_loss']:
                                plt.yscale('log')
                            else:
                                plt.ylim(top=ylim_upper,bottom=ylim_lower)

                            plt.grid()
                            plt.title(f"{attack['name']} Attack with {nb_byzantine} byz")
                            plt.legend()
                            plt.tight_layout()
                            
                            plot_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_wd_{wd}_"
                                f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_{custom_dict_to_str(attack['name'])}_comparison"
                            )
                            
                            plt.savefig(path_to_plot+"/"+plot_name + '_' + metric + '.pdf')
                            plt.close()





def paper_used_plots(path_to_results, path_to_plot, use_ref=False,
                     metric='test_accuracy', nb_steps_displayed=None,
                     path_to_results_ref = None, zoom=False
                     ):
    """
    Plot comparison of different training algorithms with their different hyperparameters on the same plot for the specified metric.
    Similar to train_loss_curve_hyperparameter but for multiple algorithms.
    """
    
    if metric not in ['test_accuracy', 'train_accuracy', 'train_loss']:
        raise ValueError("metric must be 'test_accuracy', 'train_accuracy', or 'train_loss'")
    
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")

    if use_ref:
        if path_to_results_ref is None:
            path_to_results_ref=path_to_results
        try:
            with open(os.path.join(path_to_results_ref, 'ref_config.json'), 'r') as file:
                ref_config = json.load(file)
        except Exception as e:
            print(f"ERROR reading ref_config.json: {e}")
            return

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]
    
    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]
    
    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    wd_list = data["model"]["weight_decay"]
    
    #<-------------- Training Algorithm Config ------------->
    training_algorithms = data["benchmark_config"]["training_algorithm"]
    
    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]
    
    # <-------------- Attacks Config ------------->
    attacks = data["attack"]
    
    # Ensure certain configurations are always lists
    training_algorithms = ensure_list(training_algorithms)
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)
    
    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]
    
    attacks = ensure_list(attacks)
    wd_list = ensure_list(wd_list)
    
    
    
    # Assume single values for simplicity, for the following parameters
    nb_honest = nb_honest_clients[0]

    pre_agg = pre_aggregators[0]
    pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
    pre_agg_names = "_".join(pre_agg_list_names)

    for nb_byzantine in nb_byz:
        for wd in wd_list:
            for data_dist in data_distributions:
                for agg in aggregators:
                    for dist_parameter in ensure_list(data_dist["distribution_parameter"]):
                        
                        fig, axs = plt.subplots(1, len(attacks), figsize=(10, 5), sharex=False, sharey=True)
                        if isinstance(axs,mpl.axes.Axes):
                            axs=[axs]
                        
                        if zoom:
                            axins = [inset_axes(ax, 5, 2, loc='upper center') for ax in axs]

                        if metric == 'test_accuracy' or metric == 'train_accuracy':
                            nb_points = 1 + math.ceil(nb_steps / evaluation_delta)
                            xlabel = 'Communication Rounds'
                            ylabel = 'Accuracy'
                            ylim_upper = 0.94
                            ylim_lower = 0.8 #0.9
                            file_key = 'test_accuracy_tr_seed' if metric == 'test_accuracy' else 'val_accuracy_tr_seed'
                        else:
                            nb_points = nb_steps
                            xlabel = 'Communication Rounds'
                            ylabel = 'Train Loss'
                            ylim_upper = -np.inf
                            ylim_lower = np.inf
                            file_key = 'train_loss_tr_seed'

                        ### The reference experiments are used to definie the lower value of the plot, in practice mostly for the train loss
                        if use_ref:
                            training_algorithms_ref = ensure_list(ref_config["benchmark_config"]['training_algorithm'])
                            nb_honest_ref = ensure_list(ref_config['benchmark_config']['nb_honest_clients'])[0]
                            nb_byzantine_ref = 0
                            nb_decl_ref = 0
                            for training_algorithm_dic in training_algorithms_ref:

                                training_algorithm_name = training_algorithm_dic['name']
                                lrs_ref = ensure_list(training_algorithm_dic['parameters']['learning_rate'])
                                for lr_ref in lrs_ref:
                                    momentum_ref = ensure_list(training_algorithm_dic['parameters'].get('momentum', 0.0))[0]
                                    
                                    file_ref_name = (
                                            f"{dataset_name}_{model_name}_{training_algorithm_name}_n_{nb_honest_ref}_f_{nb_byzantine_ref}_"
                                            f"d_{nb_decl_ref}_{custom_dict_to_str(data_dist['name'])}_"
                                            f"{dist_parameter}_Average_"
                                            f"_NoAttack_"
                                            f"lr_{lr_ref}_mom_{momentum_ref}_wd_{wd}"
                                        )
                                    run, run_dd = 0,0
                                    data_path_ref = os.path.join(
                                            path_to_results_ref,
                                            file_ref_name,
                                            f"{file_key}_{run + training_seed}"
                                            f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                        )
                                    tab_data_ref = genfromtxt(data_path_ref, delimiter=',')
                                    if metric in ['train_loss']:
                                        ylim_lower = np.min(tab_data_ref)
                                        
                        for attack_id, attack in enumerate(attacks):
                            nb_decl = nb_declared[0] if nb_declared[0] is not None else nb_byzantine
                            
                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                            else:
                                nb_nodes = nb_honest + nb_byzantine
                            
                            # First pass to determine y-limits, second layer in top of reference experiments, or if no ref expe
                            if True: #not use_ref:
                                for training_algorithm_dic in training_algorithms:
                                    training_algorithm = training_algorithm_dic["name"]
                                    training_algorithm_params = training_algorithm_dic.get("parameters", {})
                                    
                                    lr_list = ensure_list(training_algorithm_params.get("learning_rate", [0.01]))
                                    momentum_list = ensure_list(training_algorithm_params.get("momentum", [0.0]))
                                    

                                    for lr in lr_list:
                                        for momentum in momentum_list:
                                                tab_data = np.zeros((
                                                    nb_data_distribution_seeds,
                                                    nb_training_seeds,
                                                    nb_points
                                                ))
                                                
                                                for run_dd in range(nb_data_distribution_seeds):
                                                    for run in range(nb_training_seeds):
                                                        file_name = (
                                                            f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                            f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                            f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                            f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                            f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                        )
                                                        data_path = os.path.join(
                                                            path_to_results,
                                                            file_name,
                                                            f"{file_key}_{run + training_seed}"
                                                            f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                        )
                                                        tab_data[run_dd, run] = genfromtxt(data_path, delimiter=',')
                                                
                                                tab_data = tab_data.reshape(
                                                    nb_data_distribution_seeds * nb_training_seeds,
                                                    nb_points
                                                )
                                                                                        
                                                mean_data = np.mean(tab_data, axis = 0)
                                                
                                                if metric == "train_loss":
                                                    ylim_lower = np.min([np.min(mean_data), ylim_lower])      
                                                    ylim_upper = np.max([mean_data[0]*1.1, ylim_upper])
                                
                        
                            

                            color_index = 0
                            for training_algorithm_dic in training_algorithms:
                                training_algorithm = training_algorithm_dic["name"]
                                training_algorithm_params = training_algorithm_dic.get("parameters", {})
                                
                                lr_list = ensure_list(training_algorithm_params.get("learning_rate", [0.01]))
                                momentum_list = ensure_list(training_algorithm_params.get("momentum", [0.0]))
                                
                                for lr in lr_list:
                                    for momentum in momentum_list:
                                        
                                        tab_data = np.zeros((
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_points
                                        ))
                                        
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                file_name = (
                                                    f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                    f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                    f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                    f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                )
                                                data_path = os.path.join(
                                                    path_to_results,
                                                    file_name,
                                                    f"{file_key}_{run + training_seed}"
                                                    f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                )
                                                tab_data[run_dd, run] = genfromtxt(data_path, delimiter=',')
                                        
                                        tab_data = tab_data.reshape(
                                            nb_data_distribution_seeds * nb_training_seeds,
                                            nb_points
                                        )
                                        
                                        err = (1.96*np.std(tab_data, axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                        
                                        mean_data = np.mean(tab_data, axis = 0)

                                        if metric in ['train_loss']:
                                            mean_data = mean_data - ylim_lower
                                            if (mean_data < 1.e-6).any():
                                                print("zero :(", ylim_lower, np.min(mean_data))
                                        else:
                                            mean_data = mean_data
                                        
                                        label = f"{name_algorithms[training_algorithm]}" #+ f" lr={lr}"
                                        # if momentum > 0.0:
                                        #     label += f", ß={momentum}"
                                        
                                        axs[attack_id].plot(np.arange(nb_points)* (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1), 
                                                mean_data, label=label, color=colors[color_index % len(colors)], linestyle=tab_sign[color_index % len(tab_sign)], marker=None, alpha = 0.8, markevery=nb_points*2) #
                                        axs[attack_id].fill_between(np.arange(nb_points)* (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1), 
                                                        mean_data - err, mean_data + err, alpha=0.25)
                                        
                                        if zoom:
                                            axins[attack_id].plot(np.arange(nb_points)* (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1), 
                                                mean_data, label=label, color=colors[color_index % len(colors)], linestyle=tab_sign[color_index % len(tab_sign)], marker=None, alpha = 0.8, markevery=nb_points*2) #
                                            axins[attack_id].fill_between(np.arange(nb_points)* (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1), 
                                                        mean_data - err, mean_data + err, alpha=0.25)
                                        color_index += 1

                            
                            if zoom:
                                if metric in ['train_loss']:
                                    axins[attack_id].set_yscale('log')
                                    axins[attack_id].set_xlim(0, 100)
                                    axins[attack_id].set_yticklabels([])
                                    axins[attack_id].tick_params(axis='x',labelsize=int(size*0.8))

                                mark_inset(axs[attack_id], axins[attack_id], loc1=2,loc2=3, fc="none",ec='0.7')
                            
                            axs[attack_id].set_xlabel(xlabel)
                            axs[attack_id].set_title(name_attack[attack['name']])
                            nb_steps_lim = (nb_points-1) * (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1)
                            if nb_steps_displayed is not None:
                                nb_steps_lim = nb_steps_displayed
                            axs[attack_id].set_xlim(0, nb_steps_lim)
                            if metric in ['train_loss']:
                                axs[0].set_yscale('log')
                            else:
                                axs[0].set_ylim(top=ylim_upper,bottom=ylim_lower)
                            

                        axs[0].set_ylabel(ylabel,rotation=90,size="large")

                        handles, labels = axs[0].get_legend_handles_labels()
                        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, 0.05), ncol=5, labelspacing=0.1, handletextpad=0.1, borderaxespad=0)
                        fig.tight_layout(w_pad=0.2)

                        # plt.grid()
                        # plt.title(f"{attack['name']} Attack with {nb_byzantine} byz")
                        # plt.legend()
                        
                        plot_name = (
                            f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                            f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_wd_{wd}_"
                            f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_comparison"
                        )
                        
                        fig.savefig(path_to_plot+"/"+plot_name + '_' + metric + '.pdf',bbox_inches='tight')


def paper_used_plots_with_ref(path_to_results, path_to_plot, metric='test_accuracy', nb_steps_displayed=None,
                              path_to_results_ref=None, zoom=False):
    """
    Plot comparison of different training algorithms with their different hyperparameters on the same plot for the specified metric.
    Adds reference curves on an additional plot to the right.
    Handles different number of steps between main and reference experiments.
    """
    
    if metric not in ['test_accuracy', 'train_accuracy', 'train_loss']:
        raise ValueError("metric must be 'test_accuracy', 'train_accuracy', or 'train_loss'")
    
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")

    if path_to_results_ref is None:
        path_to_results_ref = path_to_results
    
    try:
        with open(os.path.join(path_to_results_ref, 'ref_config.json'), 'r') as file:
            ref_config = json.load(file)
    except Exception as e:
        print(f"ERROR reading ref_config.json: {e}")
        return

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_honest_clients = data["benchmark_config"]["nb_honest_clients"]
    nb_byz = data["benchmark_config"]["f"]
    nb_declared = data["benchmark_config"].get("tolerated_f", None)
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    set_honest_clients_as_clients = data["benchmark_config"]["set_honest_clients_as_clients"]
    nb_steps = data["benchmark_config"]["nb_steps"]
    
    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]
    
    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]
    wd_list = data["model"]["weight_decay"]
    
    #<-------------- Training Algorithm Config ------------->
    training_algorithms = data["benchmark_config"]["training_algorithm"]
    
    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]
    
    # <-------------- Attacks Config ------------->
    attacks = data["attack"]
    
    # Ensure certain configurations are always lists
    training_algorithms = ensure_list(training_algorithms)
    nb_honest_clients = ensure_list(nb_honest_clients)
    nb_byz = ensure_list(nb_byz)
    nb_declared = ensure_list(nb_declared)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)
    
    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]
    
    attacks = ensure_list(attacks)
    wd_list = ensure_list(wd_list)
    
    # Get reference config info
    training_algorithms_ref = ensure_list(ref_config["benchmark_config"]['training_algorithm'])
    nb_honest_ref = ensure_list(ref_config['benchmark_config']['nb_honest_clients'])[0]
    nb_steps_ref = ref_config['benchmark_config']['nb_steps']
    
    # Assume single values for simplicity, for the following parameters
    nb_honest = nb_honest_clients[0]

    pre_agg = pre_aggregators[0]
    pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
    pre_agg_names = "_".join(pre_agg_list_names)

    for nb_byzantine in nb_byz:
        for wd in wd_list:
            for data_dist in data_distributions:
                for agg in aggregators:
                    for dist_parameter in ensure_list(data_dist["distribution_parameter"]):
                        
                        if metric == 'test_accuracy' or metric == 'train_accuracy':
                            nb_points = 1 + math.ceil(nb_steps / evaluation_delta)
                            nb_points_ref = 1 + math.ceil(nb_steps_ref / evaluation_delta)
                            xlabel = 'Communication Rounds'
                            ylabel = 'Accuracy'
                            ylim_upper = 0.94
                            ylim_lower = 0.85
                            file_key = 'test_accuracy_tr_seed' if metric == 'test_accuracy' else 'val_accuracy_tr_seed'
                        else:
                            nb_points = nb_steps
                            nb_points_ref = nb_steps_ref
                            xlabel = 'Communication Rounds'
                            ylabel = 'Train Loss'
                            ylim_upper = -np.inf
                            ylim_lower = np.inf
                            file_key = 'train_loss_tr_seed'

                        # Create figure with reference curves subplot on the right
                        fig, axs = plt.subplots(1, len(attacks) + 1, figsize=(16, 5), sharex=False, sharey=True)
                        if isinstance(axs, mpl.axes.Axes):
                            axs = [axs]
                        
                        # First determine y-limits from main experiments AND reference experiments
                        for training_algorithm_dic in training_algorithms:
                            training_algorithm = training_algorithm_dic["name"]
                            training_algorithm_params = training_algorithm_dic.get("parameters", {})
                            
                            lr_list = ensure_list(training_algorithm_params.get("learning_rate", [0.01]))
                            momentum_list = ensure_list(training_algorithm_params.get("momentum", [0.0]))
                            
                            for lr in lr_list:
                                for momentum in momentum_list:
                                    for attack in attacks:
                                        nb_decl = nb_declared[0] if nb_declared[0] is not None else nb_byzantine
                                        
                                        if set_honest_clients_as_clients:
                                            nb_nodes = nb_honest
                                        else:
                                            nb_nodes = nb_honest + nb_byzantine
                                        
                                        tab_data = np.zeros((
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_points
                                        ))
                                        
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                file_name = (
                                                    f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                    f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                    f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                    f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                )
                                                data_path = os.path.join(
                                                    path_to_results,
                                                    file_name,
                                                    f"{file_key}_{run + training_seed}"
                                                    f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                )
                                                tab_data[run_dd, run] = genfromtxt(data_path, delimiter=',')
                                        
                                        tab_data = tab_data.reshape(
                                            nb_data_distribution_seeds * nb_training_seeds,
                                            nb_points
                                        )
                                        
                                        mean_data = np.mean(tab_data, axis=0)
                                        
                                        if metric == "train_loss":
                                            ylim_lower = np.min([np.min(mean_data), ylim_lower])
                                            ylim_upper = np.max([mean_data[0] * 1.1, ylim_upper])
                        
                        # Also compute y-limits from reference experiments
                        if metric == "train_loss":
                            for training_algorithm_dic in training_algorithms_ref:
                                training_algorithm_name = training_algorithm_dic['name']
                                lrs_ref = ensure_list(training_algorithm_dic['parameters']['learning_rate'])
                                
                                for lr_ref in lrs_ref:
                                    momentum_ref = ensure_list(training_algorithm_dic['parameters'].get('momentum', 0.0))[0]
                                    
                                    file_ref_name = (
                                        f"{dataset_name}_{model_name}_{training_algorithm_name}_n_{nb_honest_ref}_f_0_"
                                        f"d_0_{custom_dict_to_str(data_dist['name'])}_"
                                        f"{dist_parameter}_Average_"
                                        f"_NoAttack_"
                                        f"lr_{lr_ref}_mom_{momentum_ref}_wd_{wd}"
                                    )
                                    run, run_dd = 0, 0
                                    data_path_ref = os.path.join(
                                        path_to_results_ref,
                                        file_ref_name,
                                        f"{file_key}_{run + training_seed}"
                                        f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                    )
                                    
                                    try:
                                        tab_data_ref = genfromtxt(data_path_ref, delimiter=',')
                                        if len(tab_data_ref) > 0:
                                            ylim_lower = np.min([np.min(tab_data_ref), ylim_lower])
                                            ylim_upper = np.max([tab_data_ref[0] * 1.1, ylim_upper])
                                    except Exception:
                                        pass

                        # Build color and linestyle mapping for consistent colors and styles across subfigures
                        color_mapping = {}
                        linestyle_mapping = {}
                        color_index = 0
                        for training_algorithm_dic in training_algorithms:
                            training_algorithm = training_algorithm_dic["name"]
                            training_algorithm_params = training_algorithm_dic.get("parameters", {})
                            
                            lr_list = ensure_list(training_algorithm_params.get("learning_rate", [0.01]))
                            
                            for lr in lr_list:
                                key = (training_algorithm, lr)
                                color_mapping[key] = colors[color_index % len(colors)]
                                linestyle_mapping[key] = tab_sign[color_index % len(tab_sign)]
                                color_index += 1
                        
                        # Also add reference algorithms to the mapping
                        for training_algorithm_dic in training_algorithms_ref:
                            training_algorithm_name = training_algorithm_dic['name']
                            lrs_ref = ensure_list(training_algorithm_dic['parameters']['learning_rate'])
                            
                            for lr_ref in lrs_ref:
                                key = (training_algorithm_name, lr_ref)
                                if key not in color_mapping:
                                    color_mapping[key] = colors[color_index % len(colors)]
                                    linestyle_mapping[key] = tab_sign[color_index % len(tab_sign)]
                                    color_index += 1

                        # Plot main experiments on the left subplots
                        for attack_id, attack in enumerate(attacks):
                            nb_decl = nb_declared[0] if nb_declared[0] is not None else nb_byzantine
                            
                            if set_honest_clients_as_clients:
                                nb_nodes = nb_honest
                            else:
                                nb_nodes = nb_honest + nb_byzantine

                            for training_algorithm_dic in training_algorithms:
                                training_algorithm = training_algorithm_dic["name"]
                                training_algorithm_params = training_algorithm_dic.get("parameters", {})
                                
                                lr_list = ensure_list(training_algorithm_params.get("learning_rate", [0.01]))
                                momentum_list = ensure_list(training_algorithm_params.get("momentum", [0.0]))
                                
                                for lr in lr_list:
                                    for momentum in momentum_list:
                                        
                                        tab_data = np.zeros((
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_points
                                        ))
                                        
                                        for run_dd in range(nb_data_distribution_seeds):
                                            for run in range(nb_training_seeds):
                                                file_name = (
                                                    f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                    f"d_{nb_decl}_{custom_dict_to_str(data_dist['name'])}_"
                                                    f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                    f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                    f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                )
                                                data_path = os.path.join(
                                                    path_to_results,
                                                    file_name,
                                                    f"{file_key}_{run + training_seed}"
                                                    f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                )
                                                tab_data[run_dd, run] = genfromtxt(data_path, delimiter=',')
                                        
                                        tab_data = tab_data.reshape(
                                            nb_data_distribution_seeds * nb_training_seeds,
                                            nb_points
                                        )
                                        
                                        err = (1.96 * np.std(tab_data, axis=0)) / math.sqrt(nb_training_seeds * nb_data_distribution_seeds)
                                        
                                        mean_data = np.mean(tab_data, axis=0)

                                        if metric in ['train_loss']:
                                            mean_data = mean_data - ylim_lower
                                        
                                        label = f"{name_algorithms[training_algorithm]}" + f" lr={lr}"
                                        
                                        axs[attack_id].plot(np.arange(nb_points) * (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1), 
                                                mean_data, label=label, color=color_mapping[(training_algorithm, lr)], linestyle=linestyle_mapping[(training_algorithm, lr)], marker=None, alpha=0.8, markevery=nb_points*2)
                                        axs[attack_id].fill_between(np.arange(nb_points) * (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1), 
                                                        mean_data - err, mean_data + err, alpha=0.25)

                            axs[attack_id].set_xlabel(xlabel)
                            axs[attack_id].set_title(name_attack[attack['name']])
                            nb_steps_lim = (nb_points - 1) * (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1)
                            if nb_steps_displayed is not None:
                                nb_steps_lim = nb_steps_displayed
                            axs[attack_id].set_xlim(0, nb_steps_lim)
                            if metric in ['train_loss']:
                                axs[attack_id].set_yscale('log')
                            else:
                                axs[attack_id].set_ylim(top=ylim_upper, bottom=ylim_lower)

                        # Plot reference experiments on the rightmost subplot
                        for training_algorithm_dic in training_algorithms_ref:
                            training_algorithm_name = training_algorithm_dic['name']
                            lrs_ref = ensure_list(training_algorithm_dic['parameters']['learning_rate'])
                            
                            for lr_ref in lrs_ref:
                                momentum_ref = ensure_list(training_algorithm_dic['parameters'].get('momentum', 0.0))[0]
                                
                                file_ref_name = (
                                    f"{dataset_name}_{model_name}_{training_algorithm_name}_n_{nb_honest_ref}_f_0_"
                                    f"d_0_{custom_dict_to_str(data_dist['name'])}_"
                                    f"{dist_parameter}_Average_"
                                    f"_NoAttack_"
                                    f"lr_{lr_ref}_mom_{momentum_ref}_wd_{wd}"
                                )
                                run, run_dd = 0, 0
                                data_path_ref = os.path.join(
                                    path_to_results_ref,
                                    file_ref_name,
                                    f"{file_key}_{run + training_seed}"
                                    f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                )
                                
                                try:
                                    tab_data_ref = genfromtxt(data_path_ref, delimiter=',')
                                    
                                    # Ensure correct number of points
                                    if len(tab_data_ref) < nb_points_ref:
                                        tab_data_ref = np.pad(tab_data_ref, (0, nb_points_ref - len(tab_data_ref)), mode='edge')
                                    elif len(tab_data_ref) > nb_points_ref:
                                        tab_data_ref = tab_data_ref[:nb_points_ref]
                                    
                                    mean_data_ref = tab_data_ref
                                    
                                    if metric in ['train_loss']:
                                        mean_data_ref = mean_data_ref - ylim_lower
                                    
                                    label = f"{name_algorithms.get(training_algorithm_name, training_algorithm_name)}" + f" lr={lr_ref}"
                                    
                                    axs[-1].plot(np.arange(nb_points_ref) * (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1),
                                            mean_data_ref, label=label, color=color_mapping[(training_algorithm_name, lr_ref)], linestyle=linestyle_mapping[(training_algorithm_name, lr_ref)], marker=None, alpha=0.8, markevery=nb_points_ref*2)
                                    
                                except Exception as e:
                                    print(f"Warning: Could not load reference data from {data_path_ref}: {e}")

                        axs[-1].set_xlabel(xlabel)
                        axs[-1].set_title("Without Byzantine")
                        nb_steps_lim_ref =nb_steps_lim # (nb_points_ref - 1) * (evaluation_delta if metric in ['test_accuracy', 'train_accuracy'] else 1)
                        if nb_steps_displayed is not None:
                            nb_steps_lim_ref = nb_steps_displayed
                        axs[-1].set_xlim(0, nb_steps_lim_ref)
                        if metric in ['train_loss']:
                            axs[-1].set_yscale('log')
                            axs[-1].set_ylim(bottom=1e-3)
                        else:
                            axs[-1].set_ylim(top=ylim_upper, bottom=ylim_lower)

                        axs[0].set_ylabel(ylabel, rotation=90, size="large")

                        # Collect all unique handles and labels from all subplots
                        all_handles = []
                        all_labels = []
                        seen_labels = set()
                        
                        for ax in axs:
                            handles, labels = ax.get_legend_handles_labels()
                            for handle, label in zip(handles, labels):
                                if label not in seen_labels:
                                    all_handles.append(handle)
                                    all_labels.append(label)
                                    seen_labels.add(label)
                        
                        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.52, 0.05), ncol=5, labelspacing=0.1, handletextpad=0.1, borderaxespad=0)
                        fig.tight_layout(w_pad=0.2)
                        
                        plot_name = (
                            f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                            f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_wd_{wd}_"
                            f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_comparison_with_ref"
                        )
                        
                        fig.savefig(path_to_plot + "/" + plot_name + '_' + metric + '.pdf', bbox_inches='tight')
