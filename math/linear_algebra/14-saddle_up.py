#!/usr/bin/env python3

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication of two numpy arrays.

    Args:
        mat1 (numpy.ndarray): First input array.
        mat2 (numpy.ndarray): Second input array.

    Returns:
        numpy.ndarray: The resulting matrix product.
    """
    return np.dot(mat1, mat2)
