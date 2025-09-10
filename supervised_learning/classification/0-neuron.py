#!/usr/bin/env python3
"""
Defines a Neuron class for binary classification.
"""
import numpy as np


class Neuron:
    def __init__(self, nx):
        """
        Initializes a Neuron instance for binary classification.

        Parameters:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            TypeError: If nx is less than 1.

        Public Instance Attributes:
            nx (int): Number of input features.
            W (numpy.ndarray): Weights vector for the neuron, initialized using a normal distribution.
            b (int): Bias for the neuron, initialized to 0.
            A (int): Activated output of the neuron, initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise TypeError("nx ,must be a positive integer")
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
