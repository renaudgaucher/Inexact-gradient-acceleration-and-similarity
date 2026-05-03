import json
import os
import math

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
import seaborn as sns


size=16
size_legend=16
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.serif': 'Roman',
    'font.weight':'bold',
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

name_algorithms ={
    "DSGD":"D-NAG",
    "moDSGD":"D-GD",
    "FedProxyProx":"PIGS",
}
name_attack ={
    "Optimal_InnerProductManipulation":"IPM",
    "Optimal_ALittleIsEnough":"ALIE",
    "NoAttack":"No Attack",
    "Gaussian":"Gaussian",
    "SignFlipping":"Sign Flipping"

}

def ensure_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def load_config(path_to_results):
    """Load the config.json from an experiment directory."""
    with open(os.path.join(path_to_results, "config.json"), "r") as f:
        data = json.load(f)

    bc = data["benchmark_config"]
    model = data["model"]

    cfg = {
        "path_to_results": path_to_results,
        "training_seed": bc["training_seed"],
        "nb_training_seeds": bc["nb_training_seeds"],
        "nb_honest_clients": ensure_list(bc["nb_honest_clients"]),
        "nb_byz": ensure_list(bc["f"]),
        "nb_declared": ensure_list(bc.get("tolerated_f", None)),
        "data_distribution_seed": bc["data_distribution_seed"],
        "nb_data_distribution_seeds": bc["nb_data_distribution_seeds"],
        "data_distributions": ensure_list(bc["data_distribution"]),
        "set_honest_clients_as_clients": bc.get("set_honest_clients_as_clients", False),
        "nb_steps": bc["nb_steps"],
        "evaluation_delta": data["evaluation_and_results"]["evaluation_delta"],
        "model_name": model["name"],
        "dataset_name": model["dataset_name"],
        "weight_decay": ensure_list(model.get("weight_decay", [])),
        "training_algorithms": ensure_list(bc.get("training_algorithm", [])),
        "aggregators": ensure_list(data.get("aggregator", [])),
        "pre_aggregators": normalize_pre_aggregators(data.get("pre_aggregators", [])),
        "attacks": ensure_list(data.get("attack", [])),
    }
    return cfg


def normalize_pre_aggregators(value):
    value = ensure_list(value)
    if len(value) == 0:
        return [[]]
    if isinstance(value[0], dict):
        return [value]
    return [value]


def join_pre_agg_names(pre_agg):
    if not pre_agg:
        return ""
    out = "_".join([x.get("name", "") if isinstance(x, dict) else str(x) for x in pre_agg if x])
    return out


def custom_dict_to_str(dictionary):
    return "" if dictionary is None else str(dictionary)


def experiment_base_name(dataset_name, model_name, training_algorithm, nb_nodes,
                         nb_byzantine, nb_decl, data_dist_name, distribution_parameter,
                         agg_name, pre_agg_names,attack=None):
    out = f"{dataset_name}_{model_name}_{training_algorithm}_n_{nb_nodes}_f_{nb_byzantine}_"
    out += f"d_{nb_decl}_{custom_dict_to_str(data_dist_name)}_{distribution_parameter}_"
    out += f"{custom_dict_to_str(agg_name)}_{pre_agg_names}"
    if attack != "" and attack is not None:
         out += f"_{custom_dict_to_str(attack)}"
    return out


def experiment_base_name_from_scenario(scenario, cfg, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []
    
    training_algorithm = "" if "training_algorithm" in exclude_keys else scenario["training_algorithm"]
    nb_nodes = "" if "nb_nodes" in exclude_keys else scenario["nb_nodes"]
    nb_byzantine = "" if "nb_byzantine" in exclude_keys else scenario["nb_byzantine"]
    nb_decl = "" if "nb_decl" in exclude_keys else scenario["nb_decl"]
    data_dist_name = "" if "data_dist" in exclude_keys else scenario["data_dist"].get("name")
    distribution_parameter = "" if "distribution_parameter" in exclude_keys else scenario["distribution_parameter"]
    agg_name = "" if "agg" in exclude_keys else scenario["agg"].get("name")
    pre_agg_names = "" if "pre_agg_names" in exclude_keys else scenario["pre_agg_names"]
    attack = "" if "attack" in exclude_keys else custom_dict_to_str(scenario["attack"].get('name'))
    
    return experiment_base_name(
        cfg["dataset_name"], cfg["model_name"], training_algorithm,
        nb_nodes, nb_byzantine, nb_decl,
        data_dist_name, distribution_parameter,
        agg_name, pre_agg_names,attack,
    )


def experiment_path(path_to_results, base_name, train_or_test, run, run_dd):
    filename = f"{train_or_test}_accuracy_tr_seed_{run}_dd_seed_{run_dd}.txt"
    return os.path.join(path_to_results, base_name, filename)


def load_hyperparameters(path_to_hyperparameters, base_name, fallback):
    hyper_file = os.path.join(path_to_hyperparameters, "hyperparameters", f"{base_name}.txt")
    if os.path.exists(hyper_file):
        params = np.loadtxt(hyper_file)
        if len(params) >= 3:
            return float(params[0]), float(params[1]), float(params[2])
    return fallback


def get_nb_declared_list(nb_declared, nb_byzantine):
    if not nb_declared or nb_declared[0] is None:
        return [nb_byzantine]
    return [item for item in nb_declared if item >= nb_byzantine]


def compute_nb_accuracies(nb_steps, evaluation_delta):
    return int(1 + math.ceil(nb_steps / evaluation_delta))


def group_scenarios(scenarios, group_key):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in scenarios:
        key = s[group_key].get('name') if isinstance(s[group_key], dict) else s[group_key]
        groups[key].append(s)
        # # Create a key from all items except group_key, converting to strings for hashability
        # key_items = sorted((k, str(v)) for k, v in s.items() if k != group_key)
        # key = tuple(key_items)
        # groups[key].append(s)
    print(groups.keys())
    return list(groups.values())


def for_each_experiment(cfg):
    scenarios = []
    for training_algorithm_dic in cfg["training_algorithms"]:
        training_algorithm = training_algorithm_dic.get("name")
        training_params = training_algorithm_dic.get("parameters", {})

        lr_list = ensure_list(training_params.get("learning_rate", []))
        momentum_list = ensure_list(training_params.get("momentum", []))

        for nb_honest in cfg["nb_honest_clients"]:
            for nb_byzantine in cfg["nb_byz"]:
                nb_declared_list = get_nb_declared_list(cfg["nb_declared"], nb_byzantine)

                for nb_decl in nb_declared_list:
                    if cfg["set_honest_clients_as_clients"]:
                        nb_nodes = nb_honest
                    else:
                        nb_nodes = nb_honest + nb_byzantine

                    for data_dist in cfg["data_distributions"]:
                        for distribution_parameter in ensure_list(data_dist.get("distribution_parameter", [])):
                            for pre_agg in cfg["pre_aggregators"] or [[]]:
                                pre_agg_names = join_pre_agg_names(pre_agg)
                                for agg in cfg["aggregators"]:
                                    for attack in cfg["attacks"]:
                                        scenarios.append({
                                            "training_algorithm": training_algorithm,
                                            "lr_list": lr_list,
                                            "momentum_list": momentum_list,
                                            "nb_honest": nb_honest,
                                            "nb_byzantine": nb_byzantine,
                                            "nb_decl": nb_decl,
                                            "nb_nodes": nb_nodes,
                                            "data_dist": data_dist,
                                            "distribution_parameter": distribution_parameter,
                                            "pre_agg_names": pre_agg_names,
                                            "agg": agg,
                                            "attack": attack,
                                        })
    return scenarios

def get_name_dimension(scenario, dimension_key):
    value = scenario.get(dimension_key)
    if isinstance(value, dict):
        return value.get("name", str(value))
    return str(value)


def pretty_name(dimension_key, value):
    if isinstance(value, dict):
        value_name = value.get("name", "")
    else:
        value_name = str(value)

    if dimension_key == "attack":
        return name_attack.get(value_name, value_name)
    if dimension_key == "training_algorithm":
        return name_algorithms.get(value_name, value_name)
    return value_name


def moving_average_variable_border(y, window):
    if window <= 1:
        return y
    n = len(y)
    out = np.empty_like(y, dtype=float)
    half_left = (window - 1) // 2
    half_right = window // 2
    for i in range(n):
        start = max(0, i - half_left)
        end = min(n - 1, i + half_right)
        out[i] = y[start:end + 1].mean()
    return out


def find_best_hyperparameters(path_to_results):
    cfg = load_config(path_to_results)
    best_location = os.path.join(path_to_results, "best_hyperparameters")
    os.makedirs(os.path.join(best_location, "hyperparameters"), exist_ok=True)
    os.makedirs(os.path.join(best_location, "better_step"), exist_ok=True)

    nb_accuracies = compute_nb_accuracies(cfg["nb_steps"], cfg["evaluation_delta"])

    scenarios = for_each_experiment(cfg)
    groups = group_scenarios(scenarios, "attack")  # Group by everything except attack
    for group in groups:
        # All scenarios in group have same non-attack parameters
        scenario = group[0]  # Use first for base_name
        base_name = experiment_base_name(
            cfg["dataset_name"], cfg["model_name"], scenario["training_algorithm"],
            scenario["nb_nodes"], scenario["nb_byzantine"], scenario["nb_decl"],
            scenario["data_dist"].get("name"), scenario["distribution_parameter"],
            scenario["agg"].get("name"), scenario["pre_agg_names"],
        )

        n_combinations = (len(scenario["lr_list"]) * len(scenario["momentum_list"]) * len(cfg["weight_decay"]))
        if n_combinations == 0:
            continue

        accuracy_matrix = np.full((n_combinations, len(group)), -np.inf)  # len(group) = nb attacks
        steps_matrix = np.zeros((n_combinations, len(group)))
        hyperparameter_candidates = np.zeros((n_combinations, 3))

        idx = 0
        for lr in scenario["lr_list"]:
            for mom in scenario["momentum_list"]:
                for wd in cfg["weight_decay"]:
                    for i, scenario_attack in enumerate(group):
                        attack = scenario_attack["attack"]
                        attack_acc = np.zeros((cfg["nb_data_distribution_seeds"], cfg["nb_training_seeds"], nb_accuracies))
                        for run_dd in range(cfg["nb_data_distribution_seeds"]):
                            for run in range(cfg["nb_training_seeds"]):
                                full_name = f"{base_name}_{custom_dict_to_str(attack.get('name'))}_lr_{lr}_mom_{mom}_wd_{wd}"
                                file_path = experiment_path(path_to_results, full_name, "val", run + cfg["training_seed"], run_dd + cfg["data_distribution_seed"])
                                if os.path.exists(file_path):
                                    attack_acc[run_dd, run, :] = genfromtxt(file_path, delimiter=',')
                                else:
                                    print(f"Warning: Missing file {file_path} for hyperparameter search. Skipping this run.")
                                    attack_acc[run_dd, run, :] = - np.inf 

                        attack_acc = attack_acc.reshape(cfg["nb_data_distribution_seeds"] * cfg["nb_training_seeds"], nb_accuracies)
                        mean_acc = np.mean(attack_acc, axis=0)
                        idx_max = int(np.argmax(mean_acc))
                        accuracy_matrix[idx, i] = mean_acc[idx_max]
                        steps_matrix[idx, i] = idx_max * cfg["evaluation_delta"]

                    hyperparameter_candidates[idx, :] = [lr, mom, wd]
                    idx += 1

        if idx == 0:
            continue

        min_by_combination = np.min(accuracy_matrix[:idx, :], axis=1)
        best_idx = int(np.argmax(min_by_combination))

        best_hyp = hyperparameter_candidates[best_idx]
        best_steps = steps_matrix[best_idx, :]

        np.savetxt(os.path.join(best_location, "hyperparameters", f"{base_name}.txt"), best_hyp)

        for i, scenario_attack in enumerate(group):
            attack = scenario_attack["attack"]
            np.savetxt(
                os.path.join(best_location, "better_step", f"{base_name}_{custom_dict_to_str(attack.get('name'))}.txt"),
                np.array([best_steps[i]])
            )


def plot_metric_curve(path_to_results, path_to_plot, metric="test_accuracy", group_dimension="attack", label_dimension=None, show_lr_mom=False,
                      xlim=None, ylim=None, colors=None, markers=None, dashstyle=None, zoom_inset=False, mv_avg_window=1):
    """
    General function to plot accuracy or loss curves with flexible varying dimensions.
    
    Parameters:
    - metric: "test_accuracy", "train_accuracy", "train_loss"
    - group_dimension: dimension that vary between each plot (experiments are grouped by this dimension)
    - xlim, ylim: axis limits (tuples or None)
    - zoom_inset: whether to add a zoomed inset (for loss plots)
    """
    if colors is None:
        colors = sns.color_palette("colorblind")
    if markers is None:
        markers = ['o', 'v', 's', '*', 'd', '^', '<']
    if dashstyle is None:
        dashstyle = ['-', '--', '-.', ':', ':', 'solid']

    cfg = load_config(path_to_results)
    os.makedirs(path_to_plot, exist_ok=True)
    path_to_hyper = os.path.join(path_to_results, "best_hyperparameters")

    scenarios = for_each_experiment(cfg)
    groups = group_scenarios(scenarios, group_dimension)
    
    for group in groups:
        group_value = group[0][group_dimension].get('name') if isinstance(group[0][group_dimension], dict) else group[0][group_dimension]
        print(group_value)
        
        nb_accuracies = compute_nb_accuracies(cfg["nb_steps"], cfg["evaluation_delta"])
        # Use first scenario for common parts
        tab_data = []
        tab_x = []
        labels = []
        for scenario in group:
            # Load hyperparameters using a common base_name (exclude only attack, keep training_algorithm)
            base_name = experiment_base_name_from_scenario(scenario, cfg, exclude_keys=['attack'])
            if not show_lr_mom:
                lr, mom, wd = load_hyperparameters(path_to_hyper, base_name, (scenario["lr_list"][0] if scenario["lr_list"] else 0,
                                                                              scenario["momentum_list"][0] if scenario["momentum_list"] else 0,
                                                                              cfg["weight_decay"][0] if cfg["weight_decay"] else 0))
            else:
                lr, mom, wd = scenario["lr_list"][0], scenario["momentum_list"][0], cfg["weight_decay"][0] if cfg["weight_decay"] else 0

            scenario_data = []

            for run_dd in range(cfg["nb_data_distribution_seeds"]):
                for run in range(cfg["nb_training_seeds"]):
                    # Construct full filename with attack
                    full_name = f"{base_name}_{custom_dict_to_str(scenario["attack"].get('name'))}_lr_{lr}_mom_{mom}_wd_{wd}"
                    if metric == "test_accuracy":
                        file_path = experiment_path(path_to_results, full_name, "test", run + cfg["training_seed"], run_dd + cfg["data_distribution_seed"])
                    elif metric in ["train_accuracy", "val_accuracy"]:
                        file_path = experiment_path(path_to_results, full_name, "val", run + cfg["training_seed"], run_dd + cfg["data_distribution_seed"])
                    elif metric == "train_loss":
                        file_path = os.path.join(path_to_results, full_name, f"train_loss_tr_seed_{run + cfg['training_seed']}_dd_seed_{run_dd + cfg['data_distribution_seed']}.txt")
                    else:
                        raise ValueError(f"Unknown metric: {metric}")

                    values = genfromtxt(file_path, delimiter=',')
                    if values.ndim != 1:
                        values = values.flatten()

                    if metric == "train_loss":
                        scenario_data.append(values)
                    else:
                        # accuracy metrics align to nb_accuracies
                        if len(values) != nb_accuracies:
                            raise ValueError(f"Expected {nb_accuracies} data points for {metric}, got {len(values)} for {file_path}")
                        scenario_data.append(values)

            if metric == "train_loss":
                # use the minimum length across runs to align
                min_len = min(len(arr) for arr in scenario_data)
                aligned = np.vstack([arr[:min_len] for arr in scenario_data])
                tab_data.append(aligned)
                tab_x.append(np.arange(min_len))
            else:
                aligned = np.vstack(scenario_data)
                tab_data.append(aligned)
                tab_x.append(np.arange(nb_accuracies) * cfg["evaluation_delta"])
            
            labels.append(pretty_name(label_dimension, scenario[label_dimension]))
            if show_lr_mom:
                labels[-1] += f" (lr={lr}, mom={mom})"
        
        # Now plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (data, label) in enumerate(zip(tab_data, labels)):
            y = np.mean(data, axis=0)
            # err = (1.96 * np.std(data, axis=0)) / math.sqrt(cfg["nb_training_seeds"] * cfg["nb_data_distribution_seeds"])
            x = tab_x[i]
            if metric == "train_loss":
                y = moving_average_variable_border(y, mv_avg_window)
            ax.plot(x, y, label=label, color=colors[i % len(colors)], linestyle=dashstyle[i % len(dashstyle)], marker=None, markevery=1)# markers[i % len(markers)]
            # ax.fill_between(x, y - err, y + err, alpha=0.25)
            
        
        ax.set_xlabel("Round")
        ylabel = "Accuracy" if "accuracy" in metric else "Loss"
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(xlim)
        else:
            if metric == "train_loss":
                max_time = max(x[-1] for x in tab_x)
                ax.set_xlim(0, max_time)
            else:
                ax.set_xlim(0, (nb_accuracies - 1) * cfg["evaluation_delta"])
        if ylim:
            ax.set_ylim(ylim)
        else:
            if "accuracy" in metric:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(None, None)  # Auto for loss
        ax.grid()
        ax.legend()
        
        if zoom_inset and "loss" in metric:
            # Add zoomed inset for loss
            axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
            for i, (data, label) in enumerate(zip(tab_data, labels)):
                y = np.mean(data, axis=0)
                x = tab_x[i]
                axins.plot(x, y, color=colors[i % len(colors)], linestyle=dashstyle[i % len(dashstyle)], marker=markers[i % len(markers)], markevery=1)
            max_x = max(x[-1] for x in tab_x)
            max_y = max(np.max(np.mean(data, axis=0)) for data in tab_data)
            axins.set_xlim(0, min(max_x, 100))  # zoom to first 100 steps or full limit
            axins.set_ylim(0, min(max_y, 2))    # Example zoom at top
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        
        if True:
            plt.title(f"{pretty_name(group_dimension, scenario[group_dimension])}")
        plt.tight_layout()


        plot_name = f"{metric}_{group_value}"
        plt.savefig(os.path.join(path_to_plot, f"{plot_name}_plot.pdf"))
        plt.close()


# Backward compatibility
def test_accuracy_curve(path_to_results, path_to_plot, group_dimension="attack", colors=None, markers=None, dashstyle=None):
    plot_metric_curve(path_to_results, path_to_plot, metric="test_accuracy", group_dimension=group_dimension, 
                      colors=colors, markers=markers, dashstyle=dashstyle)


def paper_used_plots_with_ref(path_to_results, path_to_plot):
    """
    Generates the plots used in the paper, with references to specific configurations.
    This function calls the flexible plotting functions with predefined metrics and group_dimensions
    to produce the key figures from the Byzantine FL robustness analysis.

    References:
    - Figure X: Comparison of attacks under fixed parameters (test accuracy)
    - Figure Y: Comparison of training algorithms under fixed parameters (test accuracy)
    - Figure Z: Training loss curves for different algorithms (train loss with zoom)
    - Add more as needed based on paper sections
    """
    print("Generating paper plots...")

    # Figure: Accuracy curves comparing different attacks (e.g., IPM, ALIE, No Attack)
    print("Plotting attack comparisons (test accuracy)...")
    plot_metric_curve(path_to_results, path_to_plot, metric="test_accuracy", group_dimension="attack")

    # Figure: Accuracy curves comparing different training algorithms (e.g., DSGD, FedProxyProx)
    print("Plotting training algorithm comparisons (test accuracy)...")
    plot_metric_curve(path_to_results, path_to_plot, metric="test_accuracy", group_dimension="training_algorithm")

    # Figure: Training loss curves comparing different training algorithms
    print("Plotting training loss for algorithms...")
    plot_metric_curve(path_to_results, path_to_plot, metric="train_loss", group_dimension="training_algorithm", 
                      ylim=(0, 5), zoom_inset=True)  # Example limits and zoom

    # Additional plots can be added here, e.g., for aggregators or other metrics
    # For example:
    # plot_metric_curve(path_to_results, path_to_plot, metric="train_accuracy", group_dimension="agg")

    print("Paper plots generated successfully.")
