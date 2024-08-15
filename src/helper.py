"""
helper.py
~~~~~~~~~

A module containing helper functions for the MNIST handwritten digit recognition 
neural network implementation.
These functions are used across different parts of the project.
"""

import numpy as np

def sigmoid(z):
    """
    The sigmoid function.

    Parameters:
    z (numpy.ndarray): The input to the sigmoid function.

    Returns:
    numpy.ndarray: The sigmoid of the input.
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.

    Parameters:
    z (numpy.ndarray): The input to the sigmoid function.

    Returns:
    numpy.ndarray: The derivative of the sigmoid function at z.
    """
    return sigmoid(z) * (1 - sigmoid(z))