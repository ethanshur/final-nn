# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        activation_map = {
            "relu": self._relu,
            "sigmoid": self._sigmoid
        }

        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        fn = activation_map[activation]

        Z_curr = A_prev @ W_curr.T + b_curr.T
        A_curr = fn(Z_curr)
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        cache = {"A0": X}
        A_curr = X

        # Iterate through layers and call single_forward for each layer
        for idx, layer in enumerate(self.arch, start=1):
            W_curr = self._param_dict[f"W{idx}"]
            b_curr = self._param_dict[f"b{idx}"]

            A_curr, Z_curr = self._single_forward(
                W_curr=W_curr,
                b_curr=b_curr,
                A_prev=A_curr,
                activation=layer["activation"]
            )

            cache[f"Z{idx}"] = Z_curr
            cache[f"A{idx}"] = A_curr

        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        activation_backprop_map = {
            "relu": self._relu_backprop,
            "sigmoid": self._sigmoid_backprop
        }

        if activation_curr not in activation_backprop_map:
            raise ValueError(f"Unsupported activation: {activation_curr}")

        backprop_fn = activation_backprop_map[activation_curr]
        dZ_curr = backprop_fn(dA_curr, Z_curr)

        dW_curr = dZ_curr.T @ A_prev
        db_curr = np.sum(dZ_curr, axis=0, keepdims=True).T
        dA_prev = dZ_curr @ W_curr
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)

        loss_backprop_map = {
            "binary_cross_entropy": self._binary_cross_entropy_backprop,
            "bce": self._binary_cross_entropy_backprop,
            "mean_squared_error": self._mean_squared_error_backprop,
            "mse": self._mean_squared_error_backprop
        }

        if self._loss_func not in loss_backprop_map:
            raise ValueError(f"Unsupported loss function: {self._loss_func}")

        dA_curr = loss_backprop_map[self._loss_func](y, y_hat)
        grad_dict = {}

        #Iterate through layers and call single_backprop on each layer.
        for layer_idx in reversed(range(1, len(self.arch) + 1)):
            W_curr = self._param_dict[f"W{layer_idx}"]
            b_curr = self._param_dict[f"b{layer_idx}"]
            Z_curr = cache[f"Z{layer_idx}"]
            A_prev = cache[f"A{layer_idx - 1}"]
            activation_curr = self.arch[layer_idx - 1]["activation"]

            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr=W_curr,
                b_curr=b_curr,
                Z_curr=Z_curr,
                A_prev=A_prev,
                dA_curr=dA_curr,
                activation_curr=activation_curr
            )

            grad_dict[f"dW{layer_idx}"] = dW_curr
            grad_dict[f"db{layer_idx}"] = db_curr

            dA_curr = dA_prev

        return grad_dict
    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer_idx in range(1, len(self.arch) + 1):
            self._param_dict[f"W{layer_idx}"] -= self._lr * grad_dict[f"dW{layer_idx}"]
            self._param_dict[f"b{layer_idx}"] -= self._lr * grad_dict[f"db{layer_idx}"]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)
        if X_val.ndim == 1:
            X_val = X_val.reshape(1, -1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)

        loss_map = {
            "binary_cross_entropy": self._binary_cross_entropy,
            "bce": self._binary_cross_entropy,
            "mean_squared_error": self._mean_squared_error,
            "mse": self._mean_squared_error
        }

        if self._loss_func not in loss_map:
            raise ValueError(f"Unsupported loss function: {self._loss_func}")

        loss_fn = loss_map[self._loss_func]
        rng = np.random.default_rng(self._seed)

        per_epoch_loss_train = []
        per_epoch_loss_val = []

        n_samples = X_train.shape[0]

        for _ in range(self._epochs):
            permutation = rng.permutation(n_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for start_idx in range(0, n_samples, self._batch_size):
                end_idx = start_idx + self._batch_size

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_hat_batch, cache = self.forward(X_batch)
                grad_dict = self.backprop(y_batch, y_hat_batch, cache)
                self._update_params(grad_dict)

            y_hat_train, _ = self.forward(X_train)
            y_hat_val, _ = self.forward(X_val)

            per_epoch_loss_train.append(loss_fn(y_train, y_hat_train))
            per_epoch_loss_val.append(loss_fn(y_val, y_hat_val))

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)
        if self._loss_func in ("binary_cross_entropy", "bce"):
            return (y_hat >= 0.5).astype(int)

        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1/(1 + np.exp(-1*Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * self._sigmoid(Z)*(1-self._sigmoid(Z))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * (Z > 0)

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)

        term_1 = y*np.log(y_hat)
        term_2 = (1-y)*np.log(1-y_hat)

        loss = -1*(term_1 + term_2).mean()

        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)

        return (y_hat - y) / (y_hat * (1 - y_hat))

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.square(y_hat - y).mean()

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return 2 * (y_hat - y) / y.size