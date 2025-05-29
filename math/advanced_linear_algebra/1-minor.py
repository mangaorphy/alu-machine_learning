#!/usr/bin/env python3
"""
Module to calculate the minor matrix of a square matrix.
"""


def minor(matrix):
    """
    Calculate the minor matrix of a given square matrix.

    Args:
        matrix: A list of lists representing a square matrix

    Returns:
        The minor matrix (a list of lists)

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
        return [[1]]

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

    return minor_matrix


def determinant(matrix):
    """ Calculate the determinant of a square matrix.
    """
    # Input validation
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:  # Handle 0x0 matrix case
        return 1
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for N×N matrix (Laplace expansion)
    det = 0
    for col in range(n):
        # Create submatrix by excluding first row and current column
        submatrix = []
        for i in range(1, n):
            subrow = []
            for j in range(n):
                if j != col:
                    subrow.append(matrix[i][j])
            submatrix.append(subrow)

        # Calculate minor and add to determinant with alternating signs
        minor = determinant(submatrix)
        sign = (-1) ** col
        det += sign * matrix[0][col] * minor

    return det
