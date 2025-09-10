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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples

        Returns:
            The output of the neural network and the cache, respectively

        Updates:
            Updates the private attribute __cache:
            - The activated outputs of each layer are saved using key A{l}
            - X is saved to the cache using key A0
            - All neurons use sigmoid activation function

        Mathematical Process:
            For each layer l from 1 to L:
            1. Z[l] = W[l] * A[l-1] + b[l]  (linear transformation)
            2. A[l] = sigmoid(Z[l])         (activation function)

            Where sigmoid(z) = 1 / (1 + exp(-z))
        """
        # STEP 1: Store input data in cache as A0
        # A0 represents the input layer (layer 0)
        self.__cache['A0'] = X

        # STEP 2: Forward propagation through all layers using one loop
        # Start with input data as the current activation
        A_prev = X

        for l in range(1, self.__L + 1):
            # Get weights and biases for current layer
            # Shape: (current_layer_nodes, prev_layer_nodes)
            W = self.__weights[f'W{l}']
            b = self.__weights[f'b{l}']  # Shape: (current_layer_nodes, 1)

            # STEP 2.1: Linear transformation
            # Z = W * A_prev + b
            # W shape: (nodes_l, nodes_l-1), A_prev shape: (nodes_l-1, m)
            # Result Z shape: (nodes_l, m)
            Z = np.matmul(W, A_prev) + b

            # STEP 2.2: Apply sigmoid activation function
            # A = sigmoid(Z) = 1 / (1 + exp(-Z))
            A = 1 / (1 + np.exp(-Z))

            # STEP 2.3: Store activated output in cache
            self.__cache[f'A{l}'] = A

            # STEP 2.4: Update A_prev for next iteration
            A_prev = A

        # STEP 3: Return final output (last layer activation) and cache
        # The final output is the activation of the last layer (AL)
        final_output = self.__cache[f'A{self.__L}']

        return final_output, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels for input data
               m is the number of examples
               Each label should be 0 or 1
            A: numpy.ndarray with shape (1, m) containing activated output of the neuron for each example
               Each value should be between 0 and 1 (probability)

        Returns:
            The cost (a scalar value)

        Mathematical Formula:
            Cost = -1/m * Σ[Y*log(A) + (1-Y)*log(1-A)]

        Note: Uses 1.0000001 - A instead of 1 - A to avoid division by zero errors
        """
        # Get the number of examples
        m = Y.shape[1]

        # Calculate the logistic regression cost
        # Cost = -1/m * sum(Y*log(A) + (1-Y)*log(1-A))
        # Using 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the deep neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels for input data

        Returns:
            The neural network's prediction and the cost of the network, respectively
            The prediction is a numpy.ndarray with shape (1, m) containing
            the predicted labels for each example (1 if output >= 0.5, 0 otherwise)
        """
        # STEP 1: Perform forward propagation to get predictions
        A, cache = self.forward_prop(X)

        # STEP 2: Calculate cost using the output layer predictions and true
        # labels
        cost = self.cost(Y, A)

        # STEP 3: Convert probabilities to binary predictions
        # If probability >= 0.5, predict class 1; otherwise predict class 0
        predictions = np.where(A >= 0.5, 1, 0)

        # Return predictions and cost
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the deep neural network

        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels for input data
               m is the number of examples
            cache: dictionary containing all the intermediary values of the network
                   Should contain A0, A1, A2, ..., AL (activations for each layer)
            alpha: learning rate (default 0.05)

        Updates:
            Updates the private attribute __weights using backpropagation

        Mathematical Background:
            Uses backpropagation to compute gradients and update weights/biases

            For output layer (Layer L):
            dZ[L] = A[L] - Y
            dW[L] = (1/m) * dZ[L] * A[L-1]^T
            db[L] = (1/m) * sum(dZ[L])

            For hidden layers (Layer l, where l < L):
            dZ[l] = W[l+1]^T * dZ[l+1] * A[l] * (1 - A[l])  [sigmoid derivative]
            dW[l] = (1/m) * dZ[l] * A[l-1]^T
            db[l] = (1/m) * sum(dZ[l], axis=1, keepdims=True)

            Weight updates:
            W[l] = W[l] - alpha * dW[l]
            b[l] = b[l] - alpha * db[l]
        """
        # Get the number of examples
        m = Y.shape[1]

        # STEP 1: Initialize dZ for the output layer (Layer L)
        # For the output layer with sigmoid activation and cross-entropy loss:
        # dZ[L] = A[L] - Y (derivative of cost w.r.t. Z[L])
        A_L = cache[f'A{self.__L}']  # Output layer activation
        dZ = A_L - Y  # Shape: (1, m)

        # STEP 2: Backpropagation through all layers using one loop
        # Start from the output layer and work backwards
        for l in range(self.__L, 0, -1):
            # Get current layer activations and previous layer activations
            A_curr = cache[f'A{l}']      # Current layer activation
            # Previous layer activation (input to current layer)
            A_prev = cache[f'A{l - 1}']

            # STEP 2.1: Calculate gradients for weights and biases
            # dW[l] = (1/m) * dZ * A[l-1]^T
            # Shape: (nodes_l, nodes_l-1)
            dW = (1 / m) * np.matmul(dZ, A_prev.T)

            # db[l] = (1/m) * sum(dZ, axis=1, keepdims=True)
            # Shape: (nodes_l, 1)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # STEP 2.2: Update weights and biases using gradient descent
            self.__weights[f'W{l}'] = self.__weights[f'W{l}'] - alpha * dW
            self.__weights[f'b{l}'] = self.__weights[f'b{l}'] - alpha * db

            # STEP 2.3: Calculate dZ for the previous layer (if not at input
            # layer)
            if l > 1:
                # For hidden layers: dZ[l-1] = W[l]^T * dZ[l] * sigmoid_derivative(A[l-1])
                # sigmoid_derivative(A) = A * (1 - A)
                W_curr = self.__weights[f'W{l}']  # Current layer weights
                dZ = np.matmul(W_curr.T, dZ) * A_prev * \
                    (1 - A_prev)  # Shape: (nodes_l-1, m)
