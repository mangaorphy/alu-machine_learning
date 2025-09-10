#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
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
        Evaluates the neural network's predictions

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
        A1, A2 = self.forward_prop(X)

        # STEP 2: Calculate cost using the output layer predictions and true
        # labels
        cost = self.cost(Y, A2)

        # STEP 3: Convert probabilities to binary predictions
        # If probability >= 0.5, predict class 1; otherwise predict class 0
        predictions = np.where(A2 >= 0.5, 1, 0)

        # Return predictions and cost
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels for input data
            A1: numpy.ndarray with shape (nodes, m) - output of the hidden layer
            A2: numpy.ndarray with shape (1, m) - predicted output (final layer)
            alpha: learning rate (default 0.05)

        Updates:
            Updates the private attributes __W1, __b1, __W2, and __b2

        Mathematical Background:
            Uses backpropagation to compute gradients and update weights/biases

            For output layer (Layer 2):
            dZ2 = A2 - Y
            dW2 = (1/m) * dZ2 * A1^T
            db2 = (1/m) * sum(dZ2)

            For hidden layer (Layer 1):
            dZ1 = W2^T * dZ2 * A1 * (1 - A1)  [sigmoid derivative]
            dW1 = (1/m) * dZ1 * X^T
            db1 = (1/m) * sum(dZ1, axis=1, keepdims=True)
        """
        # Get the number of examples
        m = Y.shape[1]

        # STEP 1: Backpropagation for output layer (Layer 2)
        # Calculate error in output layer
        dZ2 = A2 - Y  # Shape: (1, m)

        # Calculate gradients for output layer weights and bias
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)  # Shape: (1, nodes)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # Shape: (1, 1)

        # STEP 2: Backpropagation for hidden layer (Layer 1)
        # Calculate error in hidden layer using chain rule
        # dZ1 = W2^T * dZ2 * sigmoid_derivative(A1)
        # sigmoid_derivative(A1) = A1 * (1 - A1)
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)  # Shape: (nodes, m)

        # Calculate gradients for hidden layer weights and bias
        dW1 = (1 / m) * np.matmul(dZ1, X.T)  # Shape: (nodes, nx)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # Shape: (nodes, 1)

        # STEP 3: Update weights and biases using gradient descent
        # W = W - alpha * dW, b = b - alpha * db
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

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
        Trains the neural network with enhanced monitoring and visualization capabilities

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (1, m) containing correct labels for input data
            iterations: number of iterations to train over (default 5000)
            alpha: learning rate (default 0.05)
            verbose: boolean to print training information (default True)
            graph: boolean to plot training cost (default True)
            step: step size for verbose output and graph data points (default 100)

        Returns:
            The evaluation of the training data after iterations of training

        Raises:
            TypeError: if iterations is not an integer, alpha is not a float, or step is not an integer
            ValueError: if iterations/alpha are not positive, or step is not positive and <= iterations

        Updates:
            Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2
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
        A1_initial, A2_initial = self.forward_prop(X)
        initial_cost = self.cost(Y, A2_initial)

        # Record iteration 0 data
        costs.append(initial_cost)
        iteration_points.append(0)

        # Print initial cost if verbose is enabled
        if verbose:
            print(f"Cost after 0 iterations: {initial_cost}")

        # STEP 6: Training loop (using only one loop as required)
        for i in range(1, iterations + 1):
            # Forward propagation: compute activations for current weights
            A1, A2 = self.forward_prop(X)

            # Backward propagation: update weights using gradient descent
            self.gradient_descent(X, Y, A1, A2, alpha)

            # STEP 7: Record data at specified step intervals and final
            # iteration
            if i % step == 0 or i == iterations:
                # Calculate current cost
                current_cost = self.cost(Y, A2)

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
