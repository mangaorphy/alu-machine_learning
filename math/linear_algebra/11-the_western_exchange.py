#!/usr/bin/env python3
import numpy as np


def transpose(matrix):
    """
    Transposes a 2D list (matrix) using zip(*...).

    Args:
        matrix (list of lists): Input matrix.

    Returns:
        list of lists: Transposed matrix.
    """
    return [list(row) for row in zip(*matrix)]
