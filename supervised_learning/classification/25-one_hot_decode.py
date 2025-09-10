#!/usr/bin/env python3

import numpy as np
"""
Contains function for one-hot decoding of class labels.
"""


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    Args:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
                classes is the maximum number of classes
                m is the number of examples

    Returns:
        A numpy.ndarray with shape (m,) containing the numeric labels
        for each example, or None on failure

    Mathematical Process:
        For each column (example) in the one-hot matrix:
        - Find the index of the element with value 1
        - That index represents the class label for that example

    Example:
        one_hot = [[1, 0, 0, 0],   # class 0
                   [0, 1, 0, 1],   # class 1
                   [0, 0, 1, 0]]   # class 2
        Result: [0, 1, 2, 1]
    """
    try:
        # Validate input
        if not isinstance(one_hot, np.ndarray):
            return None

        # Check if one_hot is 2D array
        if len(one_hot.shape) != 2:
            return None

        classes, m = one_hot.shape

        # Check if the matrix contains only 0s and 1s
        if not np.all((one_hot == 0) | (one_hot == 1)):
            return None

        # Check if each column has exactly one 1
        column_sums = np.sum(one_hot, axis=0)
        if not np.all(column_sums == 1):
            return None

        # Find the indices of the maximum values (which should be 1s)
        # argmax returns the index of the first occurrence of the maximum value
        labels = np.argmax(one_hot, axis=0)

        return labels

    except Exception:
        return None
