#!/usr/bin/env python3
"""
 Module to calculate the determinant of a square
 matrix using recursion.
"""


def determinant(matrix):
    """ Calculate the determinant of a square matrix.
    Args:
        matrix (list of list of int/float): A square
        matrix represented as a list of lists.
    Returns:
        int/float: The determinant of the matrix.
    Raises:
        TypeError: If the input is not a list
        of lists or if the matrix is empty.
        ValueError: If the matrix is not square
        (i.e., rows are not of equal length).
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
