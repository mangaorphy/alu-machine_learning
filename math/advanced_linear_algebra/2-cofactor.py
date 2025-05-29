#!/usr/bin/env python3
"""
Module to calculate the cofactor matrix of a square matrix.
"""


def cofactor(matrix):
    """
    Calculate the cofactor matrix of a given square matrix.

    Args:
        matrix: A list of lists representing a square matrix

    Returns:
        The cofactor matrix (a list of lists)

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Input validation
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Handle 1x1 matrix case
    if n == 1:
        return [[1]]  # Cofactor of 1x1 matrix is defined as [[1]]

    # First calculate the minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Create submatrix by excluding row i and column j
            submatrix = [row[:j] + row[j + 1:]
                         for row in (matrix[:i] + matrix[i + 1:])]

            # Calculate determinant of the submatrix
            det = determinant(submatrix)
            minor_row.append(det)
        minor_matrix.append(minor_row)

    # Then apply the sign pattern to get the cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


# Helper function - determinant calculation
def determinant(matrix):
    """Calculate determinant of a matrix (used in cofactor calculation)"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        submatrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(submatrix)
    return det
