#!/usr/bin/env python3
"""
Module the definiteness of a symmetric matrix.
"""
import numpy as np


def definiteness(matrix):
    """
    Determine the definiteness of a symmetric matrix.

    Args:
        matrix: A numpy.ndarray of shape (n, n)

    Returns:
        str: One of 'Positive definite', 'Positive semi-definite',
             'Negative semi-definite', 'Negative definite', 'Indefinite',
             or None if matrix doesn't fit any category

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Input validation
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is square and finite
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(
            np.isfinite(matrix)):
        return None

    # Check if matrix is symmetric (with some tolerance for floating point
    # errors)
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        return None

    # Calculate eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return None

    # Determine definiteness based on eigenvalues
    if np.all(eigvals > 0):
        return "Positive definite"
    elif np.all(eigvals >= 0):
        return "Positive semi-definite"
    elif np.all(eigvals < 0):
        return "Negative definite"
    elif np.all(eigvals <= 0):
        return "Negative semi-definite"
    elif np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"
    else:
        return None
