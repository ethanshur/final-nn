# TODO: import dependencies and write unit tests below
import numpy as np
from nn.nn import NeuralNetwork

from nn.preprocess import one_hot_encode_seqs, sample_seqs

import pytest

def make_simple_nn(loss_function="mean_squared_error"):
    return NeuralNetwork(
        nn_arch=[
            {"input_dim": 2, "output_dim": 2, "activation": "relu"},
            {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"},
        ],
        lr=0.01,
        seed=42,
        batch_size=2,
        epochs=2,
        loss_function=loss_function
    )
def test_single_forward():
    nn = make_simple_nn()

    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0], [0.0]])
    A_prev = np.array([[1.0, 2.0]])

    A_curr, Z_curr = nn._single_forward(W, b, A_prev, "relu")

    expected_Z = np.array([[1.0, 2.0]])
    expected_A = np.array([[1.0, 2.0]])

    assert np.allclose(Z_curr, expected_Z)
    assert np.allclose(A_curr, expected_A)

def test_forward():
    nn = make_simple_nn()

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    output, cache = nn.forward(X)

    assert output.shape == (2, 1)
    assert "A0" in cache
    assert "A1" in cache
    assert "A2" in cache
    assert "Z1" in cache
    assert "Z2" in cache

def test_single_backprop():
    nn = make_simple_nn()

    W_curr = np.array([[1.0, 0.0], [0.0, 1.0]])
    b_curr = np.array([[0.0], [0.0]])
    Z_curr = np.array([[1.0, -1.0]])
    A_prev = np.array([[2.0, 3.0]])
    dA_curr = np.array([[1.0, 1.0]])

    dA_prev, dW_curr, db_curr = nn._single_backprop(
        W_curr=W_curr,
        b_curr=b_curr,
        Z_curr=Z_curr,
        A_prev=A_prev,
        dA_curr=dA_curr,
        activation_curr="relu"
    )

    assert dA_prev.shape == A_prev.shape
    assert dW_curr.shape == W_curr.shape
    assert db_curr.shape == b_curr.shape

def test_predict():
    nn = make_simple_nn(loss_function="binary_cross_entropy")

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_pred = nn.predict(X)

    assert y_pred.shape == (2, 1)
    assert np.all(np.isin(y_pred, [0, 1]))

def test_binary_cross_entropy():
    nn = make_simple_nn(loss_function="binary_cross_entropy")

    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.9], [0.1]])

    loss = nn._binary_cross_entropy(y, y_hat)

    expected = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    assert np.isclose(loss, expected)

def test_binary_cross_entropy_backprop():
    nn = make_simple_nn(loss_function="binary_cross_entropy")

    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.9], [0.1]])

    dA = nn._binary_cross_entropy_backprop(y, y_hat)

    assert dA.shape == y.shape
    assert np.all(np.isfinite(dA))

def test_mean_squared_error():
    nn = make_simple_nn(loss_function="mean_squared_error")

    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.8], [0.2]])

    loss = nn._mean_squared_error(y, y_hat)

    expected = np.mean((y_hat - y) ** 2)
    assert np.isclose(loss, expected)

def test_mean_squared_error_backprop():
    nn = make_simple_nn(loss_function="mean_squared_error")

    y = np.array([[1.0], [0.0]])
    y_hat = np.array([[0.8], [0.2]])

    dA = nn._mean_squared_error_backprop(y, y_hat)

    expected = 2 * (y_hat - y) / y.size
    assert dA.shape == y.shape
    assert np.allclose(dA, expected)


def test_sample_seqs():
    seqs = ["AAA", "CCC", "GGG", "TTT"]
    labels = [True, False, False, False]

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    assert len(sampled_seqs) == len(sampled_labels)
    assert sum(sampled_labels) == len(sampled_labels) - sum(sampled_labels)


def test_one_hot_encode_seqs():
    seqs = ["AT", "CG"]
    encoded = one_hot_encode_seqs(seqs)

    expected = np.array([
        [1, 0, 0, 0, 0, 1, 0, 0],  # A T
        [0, 0, 1, 0, 0, 0, 0, 1],  # C G
    ])

    assert encoded.shape == (2, 8)
    assert np.array_equal(encoded, expected)
