import torch
import numpy as np

from byzfl.fed_framework import ModelBaseInterface
from byzfl.utils.conversion import flatten_dict

class Client(ModelBaseInterface):

    def __init__(self, params):
        # Check for correct types and values in params
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be of type dict, but got {type(params).__name__}")
        if not isinstance(params["loss_name"], str):
            raise TypeError(f"'loss_name' must be of type str, but got {type(params['loss_name']).__name__}")
        if not isinstance(params["LabelFlipping"], bool):
            raise TypeError(f"'LabelFlipping' must be of type bool, but got {type(params['LabelFlipping']).__name__}")
        if not isinstance(params["nb_labels"], int) or not params["nb_labels"] > 1:
            raise ValueError(f"'nb_labels' must be an integer greater than 1")
        if not isinstance(params["momentum"], float) or not 0 <= params["momentum"] < 1:
            raise ValueError(f"'momentum' must be a float in the range [0, 1)")
        if not isinstance(params["training_dataloader"], torch.utils.data.DataLoader):
            raise TypeError(f"'training_dataloader' must be a DataLoader, but got {type(params['training_dataloader']).__name__}")

        # Initialize Client instance
        super().__init__({
            # Required parameters
            "model_name": params["model_name"],
            "device": params["device"],
            # Optional parameters
            "learning_rate": params.get("learning_rate", None),
            "weight_decay": 0., # Since now l2_regularization is handled in the loss function
            "milestones": params.get("milestones", None),
            "learning_rate_decay": params.get("learning_rate_decay", None),
            "optimizer_name": params.get("optimizer_name", None),
            "optimizer_params": params.get("optimizer_params", {}),
        })

        self.criterion = getattr(torch.nn, params["loss_name"])()
        self.l2_regularization = params.get("weight_decay", 0.0)
        self.gradient_LF = 0
        self.labelflipping = params.get("LabelFlipping",False)
        self.nb_labels = params["nb_labels"]
        self.momentum = params.get("momentum",0.0)
        self.momentum_gradient = torch.zeros_like(
            torch.cat(tuple(
                tensor.view(-1) 
                for tensor in self.model.parameters()
            )),
            device=params["device"]
        )
        self.training_dataloader = params["training_dataloader"]
        self.train_iterator = iter(self.training_dataloader)
        self.store_per_client_metrics = params.get("store_per_client_metrics", False)
        self.loss_list = list()
        self.train_acc_list = list()

    def _sample_train_batch(self):
        """
        Description
        -----------
        Retrieves the next batch of data from the training dataloader. If the 
        end of the dataset is reached, the dataloader is reinitialized to start 
        from the beginning.

        Returns
        -------
        tuple
            A tuple containing the input data and corresponding target labels for the current batch.
        """
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.training_dataloader)
            return next(self.train_iterator)

    def compute_gradients(self):
        """
        Description
        -----------
        Computes the gradients of the local model's loss function for the 
        current training batch. If the `LabelFlipping` attack is enabled, 
        gradients for flipped targets are computed and stored separately. 
        Additionally, the training loss and accuracy for the batch are 
        computed and recorded.
        """
        inputs, targets = self._sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.labelflipping:
            self.model.eval()
            targets_flipped = targets.sub(self.nb_labels - 1).mul(-1)
            self._backward_pass(inputs, targets_flipped)
            self.gradient_LF = self.get_dict_gradients()
            self.model.train()

        train_loss_value = self._backward_pass(inputs, targets, train_acc=self.store_per_client_metrics)

        if self.store_per_client_metrics:
            self.loss_list.append(train_loss_value)

        return train_loss_value

    def _backward_pass(self, inputs, targets, train_acc=False):
        """
        Description
        -----------
        Performs a backward pass through the model to compute gradients for 
        the given inputs and targets. Optionally computes training accuracy 
        for the batch.

        Parameters
        ----------
        inputs : torch.Tensor
            The input data for the batch.
        targets : torch.Tensor
            The target labels for the batch.
        train_acc : bool, optional
            If True, computes and stores the training accuracy for the batch. 
            Default is False.

        Returns
        -------
        float
            The loss value for the current batch.
        """
        self.model.zero_grad()
        outputs = self.model(inputs)
        l2_reg = 0.5 * self.l2_regularization * torch.sum(torch.stack([torch.sum(p.pow(2)) for p in self.model.parameters()]))
        loss = self.criterion(outputs, targets) + l2_reg
        loss_value = loss.item()
        loss.backward()

        if train_acc:
            # Compute and store train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            acc = correct / total
            self.train_acc_list.append(acc)

        return loss_value
    
    def compute_model_update(self, num_rounds):
        """
        Description
        -----------
        Executes multiple rounds of training updates on the model. For each round,
        it samples a batch of training data, performs a backward pass to compute
        gradients, and updates the model parameters. Optionally logs training loss
        and accuracy.

        Parameters
        ----------
        num_rounds : int
            The number of training iterations to perform. Each iteration includes
            sampling a batch, computing the loss and gradients, and updating the model.

        Returns
        -------
        float
            The mean loss across all training rounds.
        """

        losses = np.zeros((num_rounds))
        for i in range(num_rounds):
            inputs, targets = self._sample_train_batch()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            train_loss_value = self._backward_pass(inputs, targets, train_acc=self.store_per_client_metrics)
            losses[i] = train_loss_value
            self.optimizer.step()

            if self.store_per_client_metrics:
                self.loss_list.append(train_loss_value)

        return losses.mean()
            

    def get_flat_flipped_gradients(self):
        """
        Description
        -----------
        Retrieves the gradients computed using flipped targets as a flat array.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            A flat array containing the gradients for the model parameters 
            when trained with flipped targets.
        """
        return flatten_dict(self.gradient_LF)

    def get_flat_gradients_with_momentum(self):
        """
        Description
        -----------
        Computes the gradients with momentum applied and returns them as a 
        flat array.

        Returns
        -------
        torch.Tensor
            A flat array containing the gradients with momentum applied.
        """
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(
            self.get_flat_gradients(),
            alpha=1 - self.momentum
        )
        return self.momentum_gradient

    def get_loss_list(self):
        """
        Description
        -----------
        Retrieves the list of training losses recorded over the course of 
        training.

        Returns
        -------
        list
            A list of float values representing the training losses for each 
            batch.
        """
        return self.loss_list

    def get_train_accuracy(self):
        """
        Description
        -----------
        Retrieves the training accuracy for each batch processed during 
        training.

        Returns
        -------
        list
            A list of float values representing the training accuracy for each 
            batch.
        """
        return self.train_acc_list

    def set_model_state(self, state_dict):
        """
        Description
        -----------
        Updates the state of the model with the provided state dictionary. 
        This method is used to load a saved model state or update 
        the global model in a federated learning context.
        Typically, this method can be used to synchronize clients with the global model.

        Parameters
        ----------
        state_dict : dict
            The state dictionary containing model parameters and buffers.

        Raises
        ------
        TypeError
            If `state_dict` is not a dictionary.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"'state_dict' must be of type dict, but got {type(state_dict).__name__}")
        self.model.load_state_dict(state_dict)



class ProxClient(Client):
    r"""
    Define a trusted client that will compute an update using 
    $$
    x_{k+1} \approx \argmin_x \langle \tilde \nabla h(x_k) + \nabla g(x_k), x \rangle + D_{g}(x;x_k)+  \frac{1}{2\eta} \|x - x_k\|^2
    $$
    where D_{g} is the Bregman divergence associated to g, the local function of client i, 
    $h = f - g$  and $\tilde \nabla h$ is the approximation of the gradient of h at point x_k issued from 
    the robust aggregation of untrusted clients' gradients at round k.
    """
    def __init__(self, params):
        """

        """
        super().__init__(params)
        self.prox_step_size = params.get("prox_step_size", 0.1)
        self.l2_regularization = params.get("weight_decay", 0.)
    

    def get_flatten_params(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()])

    def get_reference_params(self):
        return [p.detach().clone() for p in self.model.parameters()]
    
    def proximal_objective(
        self, inputs, targets,
        grad_h_tilde,   # fixed flat tensor
        ref_params     # list of tensors
    ):
        # training loss g(θ)
        preds = self.model(inputs)
        g_theta = self.criterion(preds, targets)
        if self.l2_regularization > 0.:
            # add l2 regularization of g to the prox term
            l2_reg = torch.sum(torch.stack([torch.sum(p.pow(2)) for p in self.model.parameters()]))
            g_theta += (self.l2_regularization / 2) * l2_reg

        # linear rectification <∇h̃(x_k), θ>
        theta_flat = self.get_flatten_params()
        linear_term = torch.dot(grad_h_tilde, theta_flat) 

        # (1 / 2η) ||θ - θ_k||²
        prox_term = 0.0
        for p, p0 in zip(self.model.parameters(), ref_params):
            prox_term += torch.sum((p - p0) ** 2) *0.5        

        return (linear_term + g_theta) + prox_term / self.prox_step_size

    def compute_model_prox_update(self, grad_h_tilde, verbose=False):
        """
        Description
        -----------
        Computes a proximal update of the model parameters using the provided
        approximate gradient of the non-local function. The update based on LBFGS.
        Parameters
        ----------
        grad_h_tilde : torch.Tensor
            A flat tensor representing the approximate gradient of the non-local function.
        num_rounds_max : int
            The maximum number of optimization rounds to perform for the proximal update.
        """
        # self.set_model_state(prox_parameters)
        ref_params = self.get_reference_params()
        grad_h_tilde = grad_h_tilde.detach()

        optimizer = torch.optim.LBFGS(
            self.model.parameters(), lr=0.8, history_size=500, line_search_fn='strong_wolfe',
            max_iter=2000, max_eval=20000,tolerance_grad=1e-13, tolerance_change=1e-32
            )
        
        inputs, targets = self._sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        def closure():
            optimizer.zero_grad()
            loss = self.proximal_objective(
                inputs=inputs,
                targets=targets,
                grad_h_tilde=grad_h_tilde,
                ref_params=ref_params
            )
            loss.backward()
            return loss
        
        optimizer.step(closure)

        last_loss = closure().item()
        last_grad_norm = self.get_flat_gradients().norm().item()**2.
        if verbose==1:
            optimizer.zero_grad()
            print(f"lr {self.prox_step_size} BFGS prox_loss: " + str(last_loss) + \
                      "prox gradients norm: " + str(last_grad_norm))
        
        return last_loss, last_grad_norm
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=.1, 
        # )
        
        # iter_gd = 200
        # scheduler = torch.optim.lr_scheduler.PolynomialLR(
        #         optimizer,
        #         total_iters=iter_gd,
        #         power=0.5
        #     )
        
        # inputs, targets = self._sample_train_batch()
        # inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # for i in range(iter_gd):
            
        #     optimizer.zero_grad()
        #     loss = self.proximal_objective(
        #         inputs=inputs,
        #         targets=targets,
        #         grad_h_tilde=grad_h_tilde,
        #         ref_params=ref_params
        #     )
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()
        #     if i==0 and verbose==1:
        #         print(".  Iteration " + str(i) + " prox_loss: " + str(loss.item()) + \
        #               "prox gradients norm: " + str(self.get_flat_gradients().norm().item()))
        #     if loss < 1e-8:
        #         break
        # if verbose==1:
        #     print(f"lr {self.prox_step_size}  Iteration " + str(i) + " prox_loss: " + str(loss.item()) + \
        #               "prox gradients norm: " + str(self.get_flat_gradients().norm().item()))


        
        
        
        # if verbose==1:
        #     print("Starting Proximal Update")
        #     loss = closure()
        #     print("   prox_loss: " + str(loss.item()) + "prox gradients norm: " + str(self.get_flat_gradients().norm().item()))

        # optimizer.step(closure)

        # if verbose==1:
        #     print("Finished Proximal Update...")
        #     loss = closure()
        #     print("   prox_loss: " + str(loss.item()) + "prox gradients norm: " + str(self.get_flat_gradients().norm().item()))
        