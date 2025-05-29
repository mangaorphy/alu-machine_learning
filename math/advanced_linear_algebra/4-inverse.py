#!/usr/bin/env python3
"""
Module to he inverse of a given square matrix.
"""


def inverse(matrix):
    """
    Calculate the inverse of a given square matrix.
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

    # Handle special case for 1x1 matrix
    if n == 1:
        if matrix[0][0] == 0:
            return None  # Singular matrix
        return [[1 / matrix[0][0]]]

    # Calculate determinant
    det = determinant(matrix)
    if det == 0:
        return None  # Matrix is singular

    # Calculate adjugate matrix
    adj = adjugate(matrix)

    # Calculate inverse by multiplying adjugate by 1/determinant
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adj[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix


# Helper functions
def determinant(matrix):
    """Calculate determinant of a matrix"""
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


def adjugate(matrix):
    """Calculate adjugate matrix"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            submatrix = [row[:j] + row[j + 1:]
                         for row in (matrix[:i] + matrix[i + 1:])]
            det = determinant(submatrix)
            sign = (-1) ** (i + j)
            cofactor_row.append(sign * det)
        cofactor_matrix.append(cofactor_row)

    # Transpose to get adjugate
    adjugate_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            adjugate_matrix[j][i] = cofactor_matrix[i][j]

    return adjugate_matrix
