# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import nn, preprocess


def test_forward():
    """
    Tests for proper forward pass computation
    """

    net = nn.NeuralNetwork(
        nn_arch = [{'input_dim': 2, 'output_dim': 2, 'activation': 'relu'},
                   {'input_dim': 2, 'output_dim': 1, 'activation': 'relu'}],
        lr=1e-1,
        seed=0,
        batch_size=1,
        epochs=10,
        loss_function="mse")

    net._param_dict = {"W1": np.array([[2, 2],
                                       [2, 2]]),
                       "b1": np.array([[1], [1]]),
                       "W2": np.array([[2, 2]]),
                       "b2": np.array([[1]])}
    output, cache = net.forward(np.array([10, 10]))
    assert np.array_equal(cache['A0'], np.array([10, 10]))
    assert np.array_equal(cache['A1'], np.array([[41, 41]]))
    assert np.array_equal(cache['A2'], np.array([[165]]))
    assert np.array_equal(cache['Z1'], np.array([[41, 41]]))
    assert np.array_equal(cache['Z2'], np.array([[165]]))
    assert output == 165


def test_single_forward():
    net = nn.NeuralNetwork(
        nn_arch=[{'input_dim': 5, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=1e-3,
        seed=0,
        batch_size=1,
        epochs=10,
        loss_function="mse")

    W_curr = np.array([[1, 2, 3, 4, 5]])
    b_curr = np.array([[1]])
    A_prev = np.array([[10, 10, 10, 10, 10]])

    assert net._single_forward(W_curr, b_curr, A_prev, "relu") == (np.array([[151]]), np.array([[151]]))



def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    """
    Tests for proper calculation of binary cross entropy loss.
    """
    net = nn.NeuralNetwork(
        nn_arch=[{'input_dim': 5, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=1e-3,
        seed=0,
        batch_size=1,
        epochs=10,
        loss_function="bce")

    assert np.isclose(net._loss_func(np.array([0, 1, 1]), np.array([.25, .75, .25])), 0.6538, 0.001)


def test_binary_cross_entropy_backprop():
    """
    Tests for proper calculation of binary cross entropy loss derivative.
    """
    net = nn.NeuralNetwork(
        nn_arch=[{'input_dim': 5, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=1e-3,
        seed=0,
        batch_size=1,
        epochs=10,
        loss_function="bce")
    assert np.array_equal(net._binary_cross_entropy_backprop(np.array([0, 1, 1]), np.array([.25, .75, .25])),
                           np.array([.25, -.25, -.75]))


def test_mean_squared_error():
    """
    Tests for proper calculation of mean squared error loss.
    """
    net = nn.NeuralNetwork(
        nn_arch=[{'input_dim': 5, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=1e-3,
        seed=0,
        batch_size=1,
        epochs=10,
        loss_function="mse")

    assert np.isclose(net._loss_func(np.array([4, 5, 6]), np.array([2, 2, 2])), 9.6, 0.1)


def test_mean_squared_error_backprop():
    """
    Tests for proper calculation of mean squared error loss derivative.
    """
    net = nn.NeuralNetwork(
        nn_arch=[{'input_dim': 5, 'output_dim': 1, 'activation': 'sigmoid'}],
        lr=1e-3,
        seed=0,
        batch_size=1,
        epochs=10,
        loss_function="mse")

    assert np.allclose(net._mean_squared_error_backprop(np.array([4, 5, 6]), np.array([2, 2, 2])),
                       np.array([-1.3, -2, -2.6]), 0.1)


def test_one_hot_encode():
    """
    Tests for proper one-hot encoding of DNA sequence
    """

    assert np.array_equal(preprocess.one_hot_encode_seqs(["AGTC"]),
                          np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]]))


def test_sample_seqs():
    """
    Tests for proper sampling by oversampling the underrepresented class.
    """
    seqs = ["AGG", "GGG", "AGA", "GTG", "GCG"]
    labels = [False, False, False, True, True]

    balanced_seqs, balanced_labels = preprocess.sample_seqs(seqs, labels)

    assert len(balanced_seqs) != len(labels)
    assert len(np.array(balanced_seqs)[balanced_labels]) == \
           len(np.array(balanced_seqs)[[not label for label in balanced_labels]])


