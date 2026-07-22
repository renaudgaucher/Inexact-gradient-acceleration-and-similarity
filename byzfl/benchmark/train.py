import time

import numpy as np
from torch import Tensor
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from byzfl import Client, ProxClient, Server, ByzantineClient, DataDistributor, CachedDataset, LazyCachedDataset
from byzfl.utils.misc import set_random_seed
from byzfl.benchmark.managers import ParamsManager, FileManager

transforms_hflip = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transforms_cifar_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Supported datasets
dict_datasets = {
    "mnist":        ("MNIST", transforms_mnist, transforms_mnist),
    "fashionmnist": ("FashionMNIST", transforms_hflip, transforms_hflip),
    "emnist":       ("EMNIST", transforms_mnist, transforms_mnist),
    "cifar10":      ("CIFAR10", transforms_cifar_train, transforms_cifar_test),
    "cifar100":     ("CIFAR100", transforms_cifar_train, transforms_cifar_test),
    "imagenet":     ("ImageNet", transforms_hflip, transforms_hflip)
}

### Note on the Extra-Gradient Algorithm:
# """
# Federated Extragrad Algorithm:
# Objective: minimizing f = p + q
# Algorithm ()
# 1: x_avg = tau(x_fast) + (1-tau)(x_slow)
# 2: grad_forward = nabla f(x_avg)
# 3: x_slow = argmin_x { <grad_forward, x> + (1/2 slow_lr)||x - x_slow||^2  + q(x)}
# 4: x_fast = x_fast - fast_lr * nabla f(x_slow) + momentum * (x_slow - x_avg)

# - gamma: 'momentum/fast_lr' term, in theory set as $\gamma=\mu$ (strong convexity)
# - slow_lr: proximal step size, in theory set as $slow_lr= 1/(2L_p)$
# - tau: extrapolation parameter. 
#     Theory can be set (without byz) as 
#         \tau = (slow_lr * gamma/2)**0.5
#         or \tau = fast_lr * gamma 
# - fast_lr: fast learning rate, set as $fast_lr = 1/\sqrt{4 \mu L_p)$
#     or fast_lr = (slow_lr / (2*gamma) )**0.5


# Equivalent Algorithm without division by momentum
# 1: x_avg = x_fast_tau + (1-tau)(x_slow)
# 2: grad_forward = nabla f(x_avg)
# 3: x_slow = argmin_x { <grad_forward, x> + (1/2 slow_lr)||x - x_avg||^2  + q(x)}
# 4: x_fast_tau = x_fast_tau - slow_lr/2 * nabla f(x_slow) + momentum * (tau*x_slow - x_fast_tau)
# tau*fast_lr = slow_lr/2(theory)
# \tau = momentum


# 1

# Choice of the configuration (based on the non-byzantine theory):
# 1. Tune slow_lr and momentum
# 2. set tau = momentum (/2 in the byzantine case?)
# 3. set fast_lr = slow_lr / momentum (/4 in the non byzantine case by thr)

# Note: \theta should be approximately the same as the one for ProxyProx, 
# \gamma should be approximately the strong convexity (e.g. weight decay).
# NB: gamma parametrized through 'momentum' input, for simplicity
# """

def start_training(params):
    params_manager = ParamsManager(params)

    if params.get("dtype", "float32") == "float64":
        torch.set_default_dtype(torch.float64)

    # <----------------- File Manager  ----------------->
    file_manager = FileManager({
        "result_path": params_manager.get_results_directory(),
        "dataset_name": params_manager.get_dataset_name(),
        "model_name": params_manager.get_model_name(),
        "training_algorithm_name": params_manager.get_training_algorithm_name(),
        "nb_workers": params_manager.get_nb_workers(),
        "nb_byz": params_manager.get_f(),
        "declared_nb_byz": params_manager.get_tolerated_f(),
        "data_distribution_name": params_manager.get_name_data_distribution(),
        "distribution_parameter": (
            None if params_manager.get_name_data_distribution() 
            in ["iid", "extreme_niid"] 
            else params_manager.get_parameter_data_distribution()
        ),
        "aggregation_name": params_manager.get_aggregator_name(),
        "pre_aggregation_names": [
            dict['name'] 
            for dict in params_manager.get_preaggregators()
        ],
        "attack_name": params_manager.get_attack_name(),
        "learning_rate": params_manager.get_learning_rate(),
        "momentum": params_manager.get_honest_clients_momentum(),
        "weight_decay": params_manager.get_weight_decay(),
    })

    file_manager.save_config_dict(params_manager.get_data())

    # <----------------- Federated Framework ----------------->

    # Configurations
    nb_honest_clients = params_manager.get_nb_honest_clients()
    nb_byz_clients = params_manager.get_f()
    nb_training_steps = params_manager.get_nb_steps()
    batch_size = params_manager.get_honest_clients_batch_size()

    dd_seed = params_manager.get_data_distribution_seed()
    training_seed = params_manager.get_training_seed()
    set_random_seed(dd_seed)

    training_algorithm_name = params_manager.get_training_algorithm_name()

    if training_algorithm_name not in ["DSGD", "moDSGD", "FedAvg", "FedProxyProx", "AccExtraGradProx", "AccExtraGrad"]:
        raise ValueError(f"Training algorithm {training_algorithm_name} not supported, supported algorithms are 'DSGD', 'FedAvg', and 'FedProxyProx'")
    
    if training_algorithm_name in ["FedAvg", "FedProxyProx", "AccExtraGradProx", "AccExtraGrad"]:
        training_algorithm_parameters = params_manager.get_training_algorithm_parameters()
        
        if training_algorithm_name == "FedAvg":
            proportion_selected_clients = training_algorithm_parameters["proportion_selected_clients"]
            local_steps_per_client = training_algorithm_parameters["local_steps_per_client"]
            nb_clients_to_sample = int(nb_honest_clients * proportion_selected_clients)
        else:  # FedProxyProx
            pass

    # Data Preparation
    key_dataset_name = params_manager.get_dataset_name()
    dataset_name = dict_datasets[key_dataset_name][0]
    dataset = getattr(datasets, dataset_name)(
            root = params_manager.get_data_folder(), 
            train = True, 
            download = True,
            transform = None
    )
    dataset.targets = Tensor(dataset.targets).long()

    train_size = int(params_manager.get_size_train_set() * len(dataset))
    val_size = len(dataset) - train_size

    # Split Train set into Train and Validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply transformations + use lazy caching so initialization is faster
    train_transform = dict_datasets[key_dataset_name][1]
    val_transform = dict_datasets[key_dataset_name][2]

    if params_manager.get_cache_train():
        train_dataset.dataset = LazyCachedDataset(train_dataset.dataset, transform=train_transform)
        val_dataset.dataset = LazyCachedDataset(val_dataset.dataset, transform=val_transform)

    # Prepare Validation and Test data
    
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params_manager.get_batch_size_evaluation(), 
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            persistent_workers=False
        )
    else:
        val_loader = None
    
    test_dataset = getattr(datasets, dataset_name)(
                root = params_manager.get_data_folder(),
                train=False, 
                download=True,
                transform=dict_datasets[key_dataset_name][2],
    )

    if params_manager.get_cache_test():
        test_dataset.dataset = LazyCachedDataset(test_dataset)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=params_manager.get_batch_size_evaluation(), 
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        persistent_workers=False
    )

    # Distribute data among clients using non-IID Dirichlet distribution
    data_distributor = DataDistributor({
        "data_distribution_name": params_manager.get_name_data_distribution(),
        "distribution_parameter": params_manager.get_parameter_data_distribution(),
        "nb_honest": nb_honest_clients,
        "data_loader": train_dataset,
        "batch_size": batch_size,
    })
    client_dataloaders = data_distributor.split_data()

    optimizer_params = params_manager.get_optimizer_params()
    
    momentum_clients = params_manager.get_momentum()
    if training_algorithm_name == "DSGD":
        # Set momentum in optimizer parameters
        optimizer_params["momentum"] = params_manager.get_momentum()
        momentum_clients = 0.
    elif training_algorithm_name == "moDSGD":
        # Set momentum in optimizer parameters
        optimizer_params["momentum"] = 0.


    # Initialize Honest Clients
    honest_clients = [
        Client({
            "model_name": params_manager.get_model_name(),
            "device": params_manager.get_device(),
            "optimizer_name": params_manager.get_optimizer_name(),
            "optimizer_params": optimizer_params,
            "learning_rate": params_manager.get_learning_rate(),
            "loss_name": params_manager.get_loss_name(),
            "weight_decay": params_manager.get_weight_decay(),
            "milestones": params_manager.get_milestones(),
            "learning_rate_decay": params_manager.get_learning_rate_decay(),
            "LabelFlipping": "LabelFlipping" == params_manager.get_attack_name(),
            "training_dataloader": client_dataloaders[i],
            "momentum": momentum_clients,
            "nb_labels": params_manager.get_nb_labels(),
            "store_per_client_metrics": params_manager.get_store_per_client_metrics(),
        }) for i in range(nb_honest_clients)
    ]
    if training_algorithm_name in ["AccExtraGradProx", "AccExtraGrad", "FedProxyProx"]:
        honest_clients[0] = ProxClient({
            "model_name": params_manager.get_model_name(),
            "device": params_manager.get_device(),
            "optimizer_name": params_manager.get_optimizer_name(), # Should not be needed
            "learning_rate": params_manager.get_learning_rate(),
            "loss_name": params_manager.get_loss_name(),
            "weight_decay": params_manager.get_weight_decay(),
            "milestones": params_manager.get_milestones(),
            "learning_rate_decay": params_manager.get_learning_rate_decay(), 
            "LabelFlipping": "LabelFlipping" == params_manager.get_attack_name(),
            "training_dataloader": client_dataloaders[0],
            "momentum": momentum_clients, 
            "nb_labels": params_manager.get_nb_labels(),
            "store_per_client_metrics": params_manager.get_store_per_client_metrics(),
            "prox_step_size": params_manager.get_prox_step_size(),
            "prox_optimizer_name": params_manager.get_prox_optimizer_name(),
            "prox_optimizer_params": params_manager.get_prox_optimizer_params(),
        })

    # Server Setup, Use SGD Optimizer by default
    server = Server({
        "model_name": params_manager.get_model_name(),
        "device": params_manager.get_device(),
        "validation_loader": val_loader,
        "test_loader": test_loader,
        "optimizer_name": params_manager.get_optimizer_name(),
        "optimizer_params": optimizer_params,
        "learning_rate": params_manager.get_learning_rate(),
        "weight_decay": params_manager.get_weight_decay(),
        "milestones": params_manager.get_milestones(),
        "learning_rate_decay": params_manager.get_learning_rate_decay(),
        "aggregator_info": params_manager.get_aggregator_info(),
        "pre_agg_list": params_manager.get_preaggregators(),
    })

    # Byzantine Client Setup

    attack_parameters = params_manager.get_attack_parameters()
    attack_parameters["aggregator_info"] = params_manager.get_aggregator_info()
    attack_parameters["pre_agg_list"] = params_manager.get_preaggregators()
    attack_parameters["f"] = nb_byz_clients

    # label_flipping_attack = False
    attack_name = params_manager.get_attack_name()

    label_flipping_attack = (attack_name == "LabelFlipping")

    # if label_flipping_attack and (training_algorithm_name in ["FedAvg", "FedProxyProx"]):
    #     raise ValueError(f"{training_algorithm_name} does not support Label Flipping attack.")

    attack = {
        "name": attack_name,
        "f": nb_byz_clients,
        "parameters": attack_parameters,
    }
    byz_client = ByzantineClient(attack)

    set_random_seed(training_seed)

    evaluation_delta = params_manager.get_evaluation_delta()
    evaluate_on_test = params_manager.get_evaluate_on_test()

    store_models = params_manager.get_store_models()
    store_per_client_metrics = params_manager.get_store_per_client_metrics()

    val_accuracy_list = np.array([])
    test_accuracy_list = np.array([])
    train_loss_list = np.zeros((nb_training_steps))
    prox_loss_list = np.zeros((nb_training_steps,2))

    start_time = time.time()

    def send_server_model_to_clients():
        new_model = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(new_model)
    
    def compute_clients_gradients_list(training_step, momentum=False, save=True):
        train_loss_per_client = np.zeros((nb_honest_clients))

        # Honest Clients Compute Gradients
        for i, client in enumerate(honest_clients):
            train_loss_per_client[i] = client.compute_gradients()
        
        if save:
            train_loss_list[training_step] = train_loss_per_client.mean()
        elif training_step-1 > 0:
            train_loss_list[training_step] = train_loss_list[training_step-1]
        if save and training_step==1:
            train_loss_list[0] = train_loss_list[1]

        
        # Aggregate Honest Gradients
        if momentum:
            honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]
        else:
            honest_gradients = [client.get_flat_gradients() for client in honest_clients]

        # Deal with Label Flipping Attack
        attack_input = (
            [client.get_flat_flipped_gradients() for client in honest_clients]
            if label_flipping_attack
            else honest_gradients
        )
        
        # Apply Byzantine Attack
        byz_vector = byz_client.apply_attack(attack_input)
        # Combine Honest and Byzantine Gradients
        gradients = honest_gradients + byz_vector

        return gradients, training_step + 1

    if training_algorithm_name in ["AccExtraGradProx", "AccExtraGrad"]:
        tau_factor = training_algorithm_parameters.get("tau_factor",1)
        fast_lr_factor = training_algorithm_parameters.get("fast_lr_factor",1) 
        
        slow_step_size = params_manager.get_learning_rate()
        gamma = training_algorithm_parameters.get("momentum",params_manager.get_weight_decay())   #theory: weight decay
        tau = min(1,(slow_step_size*gamma/2)**0.5*tau_factor) # max((slow_step_size/gamma/2)**0.5, 1/(training_step+1))
        fast_step_size = min(1/(2*gamma), slow_step_size/(2*tau)*fast_lr_factor)
        momentum = gamma * fast_step_size


        slow_sequence = server.get_flat_parameters() # x_f
        average_sequence = slow_sequence.clone().detach() # x_g
        fast_sequence = slow_sequence.clone().detach() # tau*x

        
        

    training_step = 0
    # Training Loop
    while training_step < nb_training_steps:

        if training_step % (max(nb_training_steps // 100, 1)) == 0:
            print(f"Training Step {training_step}/{nb_training_steps}")

        # Evaluate Global Model Every Evaluation Delta Steps
        if training_step % evaluation_delta == 0:

            if val_loader is not None:

                val_acc = server.compute_validation_accuracy()

                val_accuracy_list = np.append(val_accuracy_list, val_acc)

                file_manager.write_array_in_file(
                    val_accuracy_list, 
                    "val_accuracy_tr_seed_" + str(training_seed) 
                    + "_dd_seed_" + str(dd_seed) +".txt"
                )
            else:
                raise ValueError("Validation loader is None, cannot compute validation accuracy")

            if evaluate_on_test:
                test_acc = server.compute_test_accuracy()
                test_accuracy_list = np.append(test_accuracy_list, test_acc)

                file_manager.write_array_in_file(
                    test_accuracy_list, 
                    "test_accuracy_tr_seed_" + str(training_seed) 
                    + "_dd_seed_" + str(dd_seed) +".txt"
                )

            if store_models:
                file_manager.save_state_dict(
                    server.get_dict_parameters(),
                    training_seed,
                    dd_seed,
                    training_step
                )
        
        if training_algorithm_name == "DSGD" or training_algorithm_name == "moDSGD":
            send_server_model_to_clients()

            gradients ,training_step = compute_clients_gradients_list(training_step,momentum=True)

            # Update Global Model
            server.update_model_with_gradients(gradients)
            
        elif training_algorithm_name == "FedAvg":
            send_server_model_to_clients()

            idx_selected_clients = np.random.choice(
                range(nb_honest_clients + nb_byz_clients), 
                size=int(nb_clients_to_sample), 
                replace=False
            )

            idx_honest_clients = idx_selected_clients[idx_selected_clients < nb_honest_clients]
            count_byz_clients = len(idx_selected_clients) - len(idx_honest_clients)
            
            train_loss_per_client = np.zeros((len(idx_honest_clients)))
            honest_weights = []

            for idx, i in enumerate(idx_honest_clients):
                train_loss_per_client[idx] = honest_clients[i].compute_model_update(local_steps_per_client)
                honest_weights.append(honest_clients[i].get_flat_parameters())
            
            train_loss_list[training_step] = train_loss_per_client.mean()

            byz_client.f = count_byz_clients
            byz_weights = byz_client.apply_attack(honest_weights)

            weights = honest_weights + byz_weights

            server.update_model_with_weights(weights)
            training_step+=1
        
        elif training_algorithm_name == "FedProxyProx":
            send_server_model_to_clients()
            gradients, training_step= compute_clients_gradients_list(training_step,momentum=False)

            # Aggregate robustly the gradients on the server side using the specified robust aggregator
            grad_forward = (server.aggregate(gradients) - gradients[0]).detach() 

            prox_loss, prox_gd_norm = honest_clients[0].compute_model_prox_update(grad_forward=grad_forward,verbose=1)
            
            prox_loss_list[training_step-1,0] = prox_loss
            prox_loss_list[training_step-1,1] = prox_gd_norm
            

            # Update Global Model
            server.set_model_state(honest_clients[0].get_dict_parameters())

        elif training_algorithm_name == "AccExtraGradProx" or training_algorithm_name == "AccExtraGrad":


            #### Perform the algorithm
            average_sequence = (tau*fast_sequence + (1 - tau) * slow_sequence).clone().detach()
            slow_sequence = slow_sequence.detach()
            fast_sequence = fast_sequence.detach()
            
            server.set_parameters(average_sequence)
            send_server_model_to_clients()
            gradients, training_step = compute_clients_gradients_list(training_step, save=False, momentum=False)

            # Update the slow sequence
            if training_algorithm_name == "AccExtraGradProx": # using the prox
                grad_forward = (server.aggregate(gradients) - gradients[0]).detach()
                prox_loss, prox_gd_norm = honest_clients[0].compute_model_prox_update(grad_forward=grad_forward, verbose=1)
                prox_loss_list[training_step-1,0] = prox_loss
                prox_loss_list[training_step-1,1] = prox_gd_norm
                
                # Update Global Model
                server.set_model_state(honest_clients[0].get_dict_parameters())
            
            elif training_algorithm_name == "AccExtraGrad": # without using the prox
                server.update_model_with_gradients(gradients)
                # slow_sequence = average_sequence - slow_step_size * server.aggregate(gradients)
                # server.set_parameters(slow_sequence)
            
            slow_sequence = server.get_flat_parameters().clone().detach()

                  
            # Update the fast sequence
            send_server_model_to_clients()
            gradients, training_step = compute_clients_gradients_list(training_step,momentum=False)
            extra_gradient = server.aggregate(gradients) 
            
            fast_sequence = (fast_sequence  + momentum*(slow_sequence-fast_sequence)
                                - fast_step_size * extra_gradient).detach()

            server.set_parameters(fast_sequence)
            
            
    
        else:
            raise ValueError(f"Training algorithm {training_algorithm_name} not supported")
            
    end_time = time.time()

    file_manager.write_array_in_file(
        train_loss_list, 
        "train_loss_tr_seed_" + str(training_seed) 
        + "_dd_seed_" + str(dd_seed) +".txt"
    )

    if training_algorithm_name in ["FedProxyProx", "AccExtraGradProx"]:
        file_manager.write_array_in_file(
        prox_loss_list[:,0], 
        "prox_loss_tr_seed_" + str(training_seed) 
        + "_dd_seed_" + str(dd_seed) +".txt"
        )
        file_manager.write_array_in_file(
        prox_loss_list[:,1], 
        "prox_gd_tr_seed_" + str(training_seed) 
        + "_dd_seed_" + str(dd_seed) +".txt"
        )


    if val_loader is not None:
    
        val_acc = server.compute_validation_accuracy()

        val_accuracy_list = np.append(val_accuracy_list, val_acc)

        file_manager.write_array_in_file(
            val_accuracy_list, 
            "val_accuracy_tr_seed_" + str(training_seed) 
            + "_dd_seed_" + str(dd_seed) +".txt"
        )

    if evaluate_on_test:
        test_acc = server.compute_test_accuracy()
        test_accuracy_list = np.append(test_accuracy_list, test_acc)

        file_manager.write_array_in_file(
            test_accuracy_list, 
            "test_accuracy_tr_seed_" + str(training_seed) 
            + "_dd_seed_" + str(dd_seed) +".txt"
        )

    if store_per_client_metrics:

        for client_id, client in enumerate(honest_clients):
            loss = client.get_loss_list()
            acc = client.get_train_accuracy()
            
            file_manager.save_loss(
                loss,
                training_seed,
                dd_seed,
                client_id
            )
            
            file_manager.save_accuracy(
                acc,
                training_seed,
                dd_seed,
                client_id
            )
    
    if store_models:
        file_manager.save_state_dict(
            server.get_dict_parameters(),
            training_seed,
            dd_seed,
            training_step
        )
    
    execution_time = end_time - start_time

    file_manager.write_array_in_file(
        np.array(execution_time),
        "train_time_tr_seed_" + str(training_seed) 
        + "_dd_seed_" + str(dd_seed) +".txt"
    )
    
