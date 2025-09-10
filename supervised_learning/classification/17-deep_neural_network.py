#!/usr/bin/env python3

import numpy as np
"""
Defines a DeepNeuralNetwork class for binary classification with multiple hidden layers.
"""


class DeepNeuralNetwork:
    """
    Deep neural network performing binary classification with private attributes

    Architecture:
    Input Layer (nx features) → Hidden Layers (layers[0], layers[1], ..., layers[-1] neurons) → Output Layer (1 neuron)
    """

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network with private attributes

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each layer of the network

        Raises:
            TypeError: if nx is not an integer or layers is not a list or contains non-positive integers
            ValueError: if nx is less than 1

        Sets:
            __L: The number of layers in the neural network (private)
            __cache: A dictionary to hold all intermediary values of the network (private)
            __weights: A dictionary to hold all weights and biases of the network (private)
        """
        # STEP 1: Validate nx parameter first (order matters for exception
        # handling)
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # STEP 2: Validate layers parameter second
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # STEP 3: Validate all elements in layers are positive integers
        for layer_size in layers:
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

        # STEP 4: Set PRIVATE instance attributes

        # __L: Number of layers in the neural network
        self.__L = len(layers)

        # __cache: Dictionary to hold all intermediary values (activations)
        # Will store A0, A1, A2, ..., AL during forward propagation
        self.__cache = {}

        # __weights: Dictionary to hold all weights and biases
        # Will store W1, b1, W2, b2, ..., WL, bL
        self.__weights = {}

        # STEP 5: Initialize weights and biases using He et al. method
        # Use one loop as specified in requirements
        for l in range(1, self.__L + 1):
            # Determine input size for current layer
            if l == 1:
                # First hidden layer: input size is nx (number of input
                # features)
                input_size = nx
            else:
                # Subsequent layers: input size is the number of nodes in
                # previous layer
                input_size = layers[l - 2]

            # Current layer output size
            output_size = layers[l - 1]

            # Initialize weights using He et al. method
            # He initialization: W ~ N(0, sqrt(2/fan_in))
            # where fan_in is the number of input units in the weight tensor
            self.__weights[f'W{l}'] = np.random.randn(
                output_size, input_size) * np.sqrt(2 / input_size)

            # Initialize biases to zeros
            # Shape: (output_size, 1) to ensure proper broadcasting
            self.__weights[f'b{l}'] = np.zeros((output_size, 1))

    # Getter functions for private attributes (properties)

    @property
    def L(self):
        """Getter for the number of layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """Getter for the dictionary holding all intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the dictionary holding all weights and biases of the network"""
        return self.__weights
