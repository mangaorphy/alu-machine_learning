#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    """
    Multiplies two matrices.

    Args:
        mat1: A list of lists representing the first matrix.
        mat2: A list of lists representing the second matrix.

    Returns:
        A new matrix that is the product of mat1 and mat2, or None if the matrices cannot be multiplied.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            row.append(sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2))))
        result.append(row)

    return result
