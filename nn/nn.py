# BMI 203 Project 7: Neural Network


# Importing Dependencies
import sys

import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
#import sys
import matplotlib.pyplot as plt

# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32}, {'input_dim': 32, 'output_dim': 8}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
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
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, Union[int, str]]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        if loss_function == "mse":
            self._loss_func = self._mean_squared_error
        elif loss_function == "bce":
            self._loss_func = self._binary_cross_entropy
        else:
            print(f"Unknown loss function provided: {loss_function} \n Exiting ...")
        self.loss_function_name = loss_function

        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
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

        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T
        activation_function = self._activation_function(activation)
        return activation_function(Z_curr), Z_curr

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
        A_curr = X
        cache = {"A0": X}

        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            curr_weight_mat = self._param_dict["W" + str(layer_idx)]
            curr_bias_mat = self._param_dict["b" + str(layer_idx)]

            A_curr, Z_curr = self._single_forward(curr_weight_mat, curr_bias_mat,
                                                  A_curr, layer["activation"])
            cache["A" + str(layer_idx)] = A_curr
            cache["Z" + str(layer_idx)] = Z_curr

        return A_curr, cache

    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
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

        dA_curr_dZ = self._relu_backprop(dA_curr, Z_curr)

        delta_A = np.dot(dA_curr_dZ, W_curr)

        delta_W = dA_curr_dZ.T.dot(A_prev)

        delta_b = np.sum(dA_curr_dZ, axis=0).T[:, np.newaxis]

        return delta_A, delta_W, delta_b

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
        grad_dict = {}
        if self.loss_function_name == "mse":
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        elif self.loss_function_name == "bce":
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            raise Exception

        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1
            curr_weight_matrix = self._param_dict["W" + str(layer_idx)]
            curr_bias_matrix = self._param_dict["W" + str(layer_idx)]
            dA_curr, delta_W, delta_b = self._single_backprop(curr_weight_matrix,
                                                              curr_bias_matrix,
                                                              cache["Z" + str(layer_idx)],
                                                              cache["A" + str(layer_idx - 1)],
                                                              dA_curr,
                                                              layer["activation"])
            grad_dict["deltaW" + str(layer_idx)] = delta_W
            grad_dict["deltab" + str(layer_idx)] = delta_b

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            self._param_dict["W" + str(layer_idx)] -= grad_dict["deltaW" + str(layer_idx)] * self._lr
            self._param_dict["b" + str(layer_idx)] -= grad_dict["deltab" + str(layer_idx)] * self._lr

    # TODO
    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
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

        train_losses = []
        val_losses = []
        X_batches = [X_train[i: i + self._batch_size] for i in range(0, len(X_train), self._batch_size)]

        y_batches = [y_train[i: i + self._batch_size] for i in range(0, len(y_train), self._batch_size)]
        for epoch in range(self._epochs + 1):
            epoch_train_losses = []
            for features, labels in zip(X_batches, y_batches):
                output, cache = self.forward(features)

                train_loss = self._loss_func(labels, output)
                epoch_train_losses.append(train_loss)
                grad_dict = self.backprop(labels, output, cache)

                self._update_params(grad_dict)



            #sys.exit(0)
            train_losses.append(np.mean(epoch_train_losses))

            val_pred, _ = self.forward(X_val)
            val_loss = self._loss_func(y_val, val_pred)
            val_losses.append(val_loss)
            #print(epoch_train_losses)
            #sys.exit(0)

        return train_losses, val_losses

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        output, _ = self.forward(X)
        return output

    @staticmethod
    def _sigmoid(Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _relu(Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(Z, 0)

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
        d_sig = self._sigmoid(Z) * (1 - self._sigmoid(Z))
        return d_sig * dA

    @staticmethod
    def _relu_backprop(dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
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
        d_relu = (Z > 0).astype(int)
        # TODO
        # This seems fucked
        return d_relu * dA

    @staticmethod
    def _binary_cross_entropy(y: ArrayLike, y_hat: ArrayLike) -> float:
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

        total_loss = -(y.T.dot(np.log(y_hat)) + (1 - y.T).dot(np.log(1 - y_hat))).item()
        return total_loss / len(y)

    @staticmethod
    def _binary_cross_entropy_backprop(y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return y_hat - y # TODO WATCH OUT THIS MIGHT BE REVERSED

    @staticmethod
    def _mean_squared_error(y: ArrayLike, y_hat: ArrayLike) -> float:
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

        return np.mean((y - y_hat) ** 2).item()

    @staticmethod
    def _mean_squared_error_backprop(y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        d_mse = (2 / y.size) * (y - y_hat)
        return -d_mse

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass

    def _activation_function(self, func_name: str):
        if func_name == "relu":
            return self._relu
        elif func_name == "sigmoid":
            return self._sigmoid
        else:
            raise Exception