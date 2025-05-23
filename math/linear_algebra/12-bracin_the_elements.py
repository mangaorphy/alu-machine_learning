#!/usr/bin/env python3
"""
Performs element-wise addition, subtraction, multiplication,
and division on two numpy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division on two numpy arrays.

    Args:
        mat1 (numpy.ndarray): First input array.
        mat2 (numpy.ndarray): Second input array.

    Returns:
        tuple: A tuple containing (sum, difference, product, quotient),
        all as numpy arrays.
    """
    return (
        mat1 + mat2,
        mat1 - mat2,
        mat1 * mat2,
        mat1 / mat2
    )
