#!/usr/bin/env python3
"""
Concatenates two matrices along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specified axis.
    Args:
        mat1 (list of list of int/float): First matrix.
        mat2 (list of list of int/float): Second matrix.
        axis (int): Axis along which to concatenate
        . 0 for row-wise, 1 for column-wise.
    Returns:
        list of list of int/float: A new matrix formed
        by concatenating mat1 and mat2.
    """

    # Get dimensions
    rows1, cols1 = len(mat1), len(mat1[0])
    rows2, cols2 = len(mat2), len(mat2[0])

    # Validate axis value
    if axis not in (0, 1):
        return None

    # Check compatibility based on axis
    if axis == 0:
        if cols1 != cols2:
            return None
        # Row-wise concatenation
        result = []
        for row in mat1:
            result.append(row[:])  # copy
        for row in mat2:
            result.append(row[:])  # copy
        return result

    elif axis == 1:
        if rows1 != rows2:
            return None
        # Column-wise concatenation
        result = []
        for r in range(rows1):
            result.append(mat1[r] + mat2[r])
        return result
