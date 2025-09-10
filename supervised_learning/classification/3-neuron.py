#!/usr/bin/env python3

import numpy as np
""" Tests the Neuron class by performing forward propagation on a dataset.
"""


class Neuron:
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples

        Returns:
            The private attribute __A (activation values)
        """
        # Calculate linear combination: Z = W * X + b
        Z = np.matmul(self.__W, X) + self.__b

        # Apply sigmoid activation function: A = 1 / (1 + e^(-Z))
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels for input data
               m is the number of examples
            A: numpy.ndarray with shape (1, m) containing activated output of the neuron for each example
        Returns:
            The cost
        """
        # Get the number of examples
        m = Y.shape[1]

        # Calculate the logistic regression cost
        # Cost = -1/m * sum(Y*log(A) + (1-Y)*log(1-A))
        # Using 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost
