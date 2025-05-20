#!/usr/bin/env python3

def cat_matrices2D(mat1, mat2, axis=0):

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
