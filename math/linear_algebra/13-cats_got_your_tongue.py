#!/usr/bin/env python3
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two numpy arrays along a specified axis.

    Args:
        mat1 (numpy.ndarray): First input array.
        mat2 (numpy.ndarray): Second input array.
        axis (int): Axis along which to concatenate. Default is 0.

    Returns:
        numpy.ndarray: A new array formed by concatenating mat1 and mat2.
    """
    return np.concatenate((mat1, mat2), axis=axis)
