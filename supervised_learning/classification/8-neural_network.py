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
        Initialize the neural network

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

        # Initialize weights and biases for the hidden layer (layer 1)
        # W1 shape: (nodes, nx) - each row represents weights for one hidden neuron
        # W1[i,j] = weight from input feature j to hidden neuron i
        self.W1 = np.random.randn(nodes, nx)

        # b1 shape: (nodes, 1) - bias for each hidden neuron
        # Initialize with zeros as specified
        self.b1 = np.zeros((nodes, 1))

        # A1 shape: (nodes, m) where m is number of examples
        # Activated output of hidden layer (will be set during forward
        # propagation)
        self.A1 = 0

        # Initialize weights and biases for the output layer (layer 2)
        # W2 shape: (1, nodes) - weights from hidden layer to output neuron
        # W2[0,i] = weight from hidden neuron i to output neuron
        self.W2 = np.random.randn(1, nodes)

        # b2 shape: (1, 1) - bias for the output neuron
        # Initialize with zero as specified
        self.b2 = 0

        # A2 shape: (1, m) where m is number of examples
        # Activated output of output layer (final prediction)
        self.A2 = 0
