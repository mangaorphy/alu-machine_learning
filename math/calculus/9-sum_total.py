#!/usr/bin/env python3
"""
Module to calculate the sum of squares from 1 to n (Σi²)
"""


def summation_i_squared(n):
    """
    Calculate the sum of squares from 1 to n (Σi²)

    Args:
        n (int): The stopping condition (must be positive integer)

    Returns:
        int: The sum of squares if n is valid, None otherwise
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
