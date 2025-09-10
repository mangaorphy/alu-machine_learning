#!/usr/bin/env python3

import numpy as np
"""
Contains function for one-hot encoding of class labels.
"""


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels
           m is the number of examples
        classes: maximum number of classes found in Y

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure

    Mathematical Process:
        For each example i with label Y[i]:
        - Create a vector of zeros with length 'classes'
        - Set the Y[i]-th element to 1
        - Stack all vectors to form the one-hot matrix

    Example:
        Y = [0, 1, 2, 1], classes = 3
        Result:
        [[1, 0, 0, 0],   # class 0
         [0, 1, 0, 1],   # class 1
         [0, 0, 1, 0]]   # class 2
    """
    try:
        # Validate inputs
        if not isinstance(Y, np.ndarray):
            return None

        if not isinstance(classes, int):
            return None

        if classes <= 0:
            return None

        # Check if Y is 1D array
        if len(Y.shape) != 1:
            return None

        # Get number of examples
        m = Y.shape[0]

        # Check if all labels are valid (non-negative integers less than
        # classes)
        if not np.issubdtype(Y.dtype, np.integer):
            return None

        if np.any(Y < 0) or np.any(Y >= classes):
            return None

        # Create one-hot encoding matrix
        # Shape: (classes, m)
        one_hot = np.zeros((classes, m))

        # Set appropriate elements to 1
        # For each example i, set one_hot[Y[i], i] = 1
        one_hot[Y, np.arange(m)] = 1

        return one_hot

    except Exception:
        return None
