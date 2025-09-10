#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
"""
Defines a DeepNeuralNetwork class for multiclass classification with multiple hidden layers and configurable activation functions.
"""


class DeepNeuralNetwork:
    """
    Deep neural network performing multiclass classification with private attributes and configurable activation functions

    Architecture:
    Input Layer (nx features) → Hidden Layers (layers[0], layers[1], ..., layers[-2] neurons) → Output Layer (layers[-1] neurons)
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize the deep neural network with private attributes and configurable activation function

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each layer of the network
            activation: type of activation function used in the hidden layers ('sig' or 'tanh')

        Raises:
            TypeError: if nx is not an integer or layers is not a list or contains non-positive integers
            ValueError: if nx is less than 1 or activation is not 'sig' or 'tanh'

        Sets:
            __L: The number of layers in the neural network (private)
            __cache: A dictionary to hold all intermediary values of the network (private)
            __weights: A dictionary to hold all weights and biases of the network (private)
            __activation: The activation function type for hidden layers (private)
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

        # STEP 4: Validate activation parameter
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        # STEP 5: Set PRIVATE instance attributes

        # __L: Number of layers in the neural network
        self.__L = len(layers)

        # __cache: Dictionary to hold all intermediary values (activations)
        # Will store A0, A1, A2, ..., AL during forward propagation
        self.__cache = {}

        # __weights: Dictionary to hold all weights and biases
        # Will store W1, b1, W2, b2, ..., WL, bL
        self.__weights = {}

        # __activation: The activation function type for hidden layers
        self.__activation = activation

        # STEP 6: Initialize weights and biases using He et al. method
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

    @property
    def activation(self):
        """Getter for the activation function type used in hidden layers"""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the deep neural network for multiclass classification
        with configurable activation functions for hidden layers

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
            - Hidden layers use the specified activation function (sigmoid or tanh)
            - Output layer uses softmax activation function for multiclass classification

        Mathematical Process:
            For hidden layers l from 1 to L-1:
            1. Z[l] = W[l] * A[l-1] + b[l]  (linear transformation)
            2. A[l] = activation_function(Z[l])  (sigmoid or tanh activation function)

            For output layer L:
            1. Z[L] = W[L] * A[L-1] + b[L]  (linear transformation)
            2. A[L] = softmax(Z[L])         (softmax activation function)

            Where:
            - sigmoid(z) = 1 / (1 + exp(-z))
            - tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
            - softmax(z_i) = exp(z_i) / sum(exp(z_j)) for all j
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

            # STEP 2.2: Apply activation function
            if l == self.__L:
                # Output layer: use softmax activation for multiclass classification
                # Softmax: A_i = exp(Z_i) / sum(exp(Z_j)) for all j
                # For numerical stability, subtract max from Z before computing
                # exp
                Z_stable = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_stable)
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Hidden layers: use the specified activation function
                if self.__activation == 'sig':
                    # Sigmoid activation: A = 1 / (1 + exp(-Z))
                    A = 1 / (1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    # Tanh activation: A = tanh(Z)
                    A = np.tanh(Z)

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
        Calculates the cost of the model using categorical cross-entropy for multiclass classification

        Args:
            Y: numpy.ndarray with shape (classes, m) containing one-hot encoded correct labels
               classes is the number of classes
               m is the number of examples
            A: numpy.ndarray with shape (classes, m) containing activated output (probabilities) for each example
               Each column should sum to 1 (softmax output)

        Returns:
            The cost (a scalar value)

        Mathematical Formula:
            Cost = -1/m * Σ Σ Y[i,j] * log(A[i,j])
            where i ranges over classes and j ranges over examples

        Note: Uses np.clip to avoid log(0) errors by ensuring A values are in range [1e-7, 1-1e-7]
        """
        # Get the number of examples
        m = Y.shape[1]

        # Clip A to avoid log(0) which would result in -inf
        # Ensure all values are in range [1e-7, 1-1e-7]
        A_clipped = np.clip(A, 1e-7, 1 - 1e-7)

        # Calculate the categorical cross-entropy cost
        # Cost = -1/m * sum(Y * log(A))
        cost = -1 / m * np.sum(Y * np.log(A_clipped))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the deep neural network's predictions for multiclass classification

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (classes, m) containing one-hot encoded correct labels

        Returns:
            The neural network's prediction and the cost of the network, respectively
            The prediction is a numpy.ndarray with shape (classes, m) containing
            the one-hot encoded predicted labels for each example
        """
        # STEP 1: Perform forward propagation to get predictions
        A, cache = self.forward_prop(X)

        # STEP 2: Calculate cost using the output layer predictions and true
        # labels
        cost = self.cost(Y, A)

        # STEP 3: Convert probabilities to one-hot encoded predictions
        # Find the class with highest probability for each example
        # argmax returns the index of the maximum value along axis 0
        predicted_classes = np.argmax(A, axis=0)  # Shape: (m,)

        # Convert to one-hot encoding
        # Create a matrix of zeros with same shape as A
        predictions = np.zeros_like(A)
        # Set the predicted class to 1 for each example
        predictions[predicted_classes, np.arange(A.shape[1])] = 1

        # Return predictions and cost
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the deep neural network
        with support for different activation functions in hidden layers

        Args:
            Y: numpy.ndarray with shape (classes, m) containing one-hot encoded correct labels
               m is the number of examples
            cache: dictionary containing all the intermediary values of the network
                   Should contain A0, A1, A2, ..., AL (activations for each layer)
            alpha: learning rate (default 0.05)

        Updates:
            Updates the private attribute __weights using backpropagation

        Mathematical Background:
            Uses backpropagation to compute gradients and update weights/biases

            For output layer (Layer L) with softmax and categorical cross-entropy:
            dZ[L] = A[L] - Y
            dW[L] = (1/m) * dZ[L] * A[L-1]^T
            db[L] = (1/m) * sum(dZ[L])

            For hidden layers (Layer l, where l < L):
            - If activation is 'sig' (sigmoid): derivative = A[l] * (1 - A[l])
            - If activation is 'tanh': derivative = 1 - A[l]^2

            dZ[l] = W[l+1]^T * dZ[l+1] * activation_derivative(A[l])
            dW[l] = (1/m) * dZ[l] * A[l-1]^T
            db[l] = (1/m) * sum(dZ[l], axis=1, keepdims=True)

            Weight updates:
            W[l] = W[l] - alpha * dW[l]
            b[l] = b[l] - alpha * db[l]
        """
        # Get the number of examples
        m = Y.shape[1]

        # STEP 1: Initialize dZ for the output layer (Layer L)
        # For the output layer with softmax activation and categorical cross-entropy loss:
        # dZ[L] = A[L] - Y (derivative of cost w.r.t. Z[L])
        A_L = cache[f'A{self.__L}']  # Output layer activation
        dZ = A_L - Y  # Shape: (classes, m)

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
                # For hidden layers: dZ[l-1] = W[l]^T * dZ[l] *
                # activation_derivative(A[l-1])
                W_curr = self.__weights[f'W{l}']  # Current layer weights

                # Calculate activation derivative based on the activation
                # function
                if self.__activation == 'sig':
                    # Sigmoid derivative: sigmoid'(z) = sigmoid(z) * (1 -
                    # sigmoid(z)) = A * (1 - A)
                    activation_derivative = A_prev * (1 - A_prev)
                elif self.__activation == 'tanh':
                    # Tanh derivative: tanh'(z) = 1 - tanh(z)^2 = 1 - A^2
                    activation_derivative = 1 - A_prev ** 2

                # Shape: (nodes_l-1, m)
                dZ = np.matmul(W_curr.T, dZ) * activation_derivative

    def train(
            self,
            X,
            Y,
            iterations=5000,
            alpha=0.05,
            verbose=True,
            graph=True,
            step=100):
        """
        Trains the deep neural network with enhanced monitoring and visualization capabilities

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (classes, m) containing one-hot encoded correct labels
            iterations: number of iterations to train over (default 5000)
            alpha: learning rate (default 0.05)
            verbose: boolean to print training information (default True)
            graph: boolean to plot training cost (default True)
            step: step size for verbose output and graph data points (default 100)

        Returns:
            The evaluation of the training data after iterations of training have occurred

        Raises:
            TypeError: if iterations is not an integer, alpha is not a float, or step is not an integer
            ValueError: if iterations/alpha are not positive, or step is not positive and <= iterations

        Updates:
            Updates the private attributes __weights and __cache
        """
        # STEP 1: Validate iterations parameter (must come first)
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        # STEP 2: Validate alpha parameter (must come second)
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # STEP 3: Validate step parameter (only if verbose or graph is True)
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # STEP 4: Initialize lists for tracking cost (for graphing)
        costs = []
        iteration_points = []

        # STEP 5: Record initial state (iteration 0)
        # Calculate initial cost before any training
        A_initial, cache_initial = self.forward_prop(X)
        initial_cost = self.cost(Y, A_initial)

        # Record iteration 0 data
        costs.append(initial_cost)
        iteration_points.append(0)

        # Print initial cost if verbose is enabled
        if verbose:
            print(f"Cost after 0 iterations: {initial_cost}")

        # STEP 6: Training loop (using only one loop as required)
        for i in range(1, iterations + 1):
            # Forward propagation: compute activations and update cache
            A, cache = self.forward_prop(X)

            # Backward propagation: update weights using gradient descent
            self.gradient_descent(Y, cache, alpha)

            # STEP 7: Record data at specified step intervals and final
            # iteration
            if i % step == 0 or i == iterations:
                # Calculate current cost
                current_cost = self.cost(Y, A)

                # Store for graphing
                costs.append(current_cost)
                iteration_points.append(i)

                # Print progress if verbose is enabled
                if verbose:
                    print(f"Cost after {i} iterations: {current_cost}")

        # STEP 8: Create training cost graph if graph is enabled
        if graph:
            plt.figure(figsize=(10, 6))
            plt.plot(iteration_points, costs, 'b-', linewidth=2)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.grid(True, alpha=0.3)
            plt.show()

        # STEP 9: Return final evaluation after all training iterations
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
            filename: the file to which the object should be saved
                     If filename does not have the extension .pkl, it will be added

        Process:
            - Adds .pkl extension if not present
            - Uses pickle to serialize the entire object
            - Saves to the specified file path
        """
        # Check if filename already has .pkl extension
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'

        # Save the object using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
            filename: the file from which the object should be loaded

        Returns:
            The loaded object, or None if filename doesn't exist

        Process:
            - Checks if file exists
            - Uses pickle to deserialize the object
            - Returns the loaded DeepNeuralNetwork instance
        """
        try:
            # Check if file exists
            if not os.path.exists(filename):
                return None

            # Load the object using pickle
            with open(filename, 'rb') as f:
                loaded_object = pickle.load(f)

            return loaded_object

        except Exception:
            # Return None if any error occurs during loading
            return None
