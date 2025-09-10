#!/usr/bin/env python3

import numpy as np
"""
Defines a NeuralNetwork class for binary classification with one hidden layer.
"""


class NeuralNetwork:
    """
    Neural network with one hidden layer performing binary classification

    Architecture:
    Input Layer (nx features) → Hidden Layer (nodes neurons) → Output Layer (1 neuron)
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network with private attributes

        Args:
            nx: number of input features
            nodes: number of nodes in the hidden layer

        Raises:
            TypeError: if nx or nodes is not an integer
            ValueError: if nx or nodes is less than 1
        """
        # Validate nx parameter first (order matters for exception handling)
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes parameter second
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize PRIVATE weights and biases for the hidden layer (layer 1)
        # __W1 shape: (nodes, nx) - each row represents weights for one hidden neuron
        # __W1[i,j] = weight from input feature j to hidden neuron i
        self.__W1 = np.random.randn(nodes, nx)

        # __b1 shape: (nodes, 1) - bias for each hidden neuron
        # Initialize with zeros as specified
        self.__b1 = np.zeros((nodes, 1))

        # __A1 shape: (nodes, m) where m is number of examples
        # Activated output of hidden layer (will be set during forward
        # propagation)
        self.__A1 = 0

        # Initialize PRIVATE weights and biases for the output layer (layer 2)
        # __W2 shape: (1, nodes) - weights from hidden layer to output neuron
        # __W2[0,i] = weight from hidden neuron i to output neuron
        self.__W2 = np.random.randn(1, nodes)

        # __b2 shape: (1, 1) - bias for the output neuron
        # Initialize with zero as specified
        self.__b2 = 0

        # __A2 shape: (1, m) where m is number of examples
        # Activated output of output layer (final prediction)
        self.__A2 = 0

    # Getter functions for private attributes (properties)

    @property
    def W1(self):
        """Getter for hidden layer weights"""
        return self.__W1

    @property
    def b1(self):
        """Getter for hidden layer bias"""
        return self.__b1

    @property
    def A1(self):
        """Getter for hidden layer activated output"""
        return self.__A1

    @property
    def W2(self):
        """Getter for output layer weights"""
        return self.__W2

    @property
    def b2(self):
        """Getter for output layer bias"""
        return self.__b2

    @property
    def A2(self):
        """Getter for output layer activated output (prediction)"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples

        Returns:
            The private attributes __A1 and __A2, respectively

        Updates:
            Updates the private attributes __A1 and __A2
        """
        # STEP 1: Forward propagation through hidden layer (Layer 1)
        # Calculate linear combination: Z1 = W1 * X + b1
        # W1 shape: (nodes, nx), X shape: (nx, m) → Z1 shape: (nodes, m)
        Z1 = np.matmul(self.__W1, X) + self.__b1

        # Apply sigmoid activation function: A1 = sigmoid(Z1)
        # A1 shape: (nodes, m) - activated output of hidden layer
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # STEP 2: Forward propagation through output layer (Layer 2)
        # Calculate linear combination: Z2 = W2 * A1 + b2
        # W2 shape: (1, nodes), A1 shape: (nodes, m) → Z2 shape: (1, m)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2

        # Apply sigmoid activation function: A2 = sigmoid(Z2)
        # A2 shape: (1, m) - final prediction (probability of positive class)
        self.__A2 = 1 / (1 + np.exp(-Z2))

        # Return both hidden layer and output layer activations
        return self.__A1, self.__A2
