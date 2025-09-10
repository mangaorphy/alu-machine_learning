#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
"""
Defines a Neuron class for binary classification.
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

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels for input data

        Returns:
            The neuron's prediction and the cost of the network, respectively
            The prediction is a numpy.ndarray with shape (1, m) containing
            the predicted labels for each example (1 if output >= 0.5, 0 otherwise)
        """
        # Forward propagation to get activations
        A = self.forward_prop(X)

        # Calculate cost
        cost = self.cost(Y, A)

        # Convert activations to predictions (1 if A >= 0.5, 0 otherwise)
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
               nx is the number of input features to the neuron
               m is the number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels for input data
            A: numpy.ndarray with shape (1, m) containing activated output of the neuron for each example
            alpha: the learning rate (default 0.05)

        Updates the private attributes __W and __b
        """
        # Get the number of examples
        m = Y.shape[1]

        # Calculate the gradients
        # dW = (1/m) * X * (A - Y)^T
        dW = (1 / m) * np.matmul(X, (A - Y).T)

        # db = (1/m) * sum(A - Y)
        db = (1 / m) * np.sum(A - Y)

        # Update weights and bias using gradient descent
        self.__W = self.__W - alpha * dW.T
        self.__b = self.__b - alpha * db

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
        Trains the neuron with enhanced monitoring and visualization capabilities

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
        A_initial = self.forward_prop(X)
        initial_cost = self.cost(Y, A_initial)

        # Record iteration 0 data
        costs.append(initial_cost)
        iteration_points.append(0)

        # Print initial cost if verbose is enabled
        if verbose:
            print(f"Cost after 0 iterations: {initial_cost}")

        # STEP 6: Training loop
        for i in range(1, iterations + 1):
            # Forward propagation: calculate activations
            A = self.forward_prop(X)

            # Gradient descent: update weights and bias
            self.gradient_descent(X, Y, A, alpha)

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

        # STEP 9: Return final evaluation
        return self.evaluate(X, Y)
