#!/usr/bin/env python3
"""Forward Propagation"""

import tensorflow as tf
create_layer = __import__("1-create_layer").create_layer

def forward_prop(x, layer_sizes=[], activation=[]):
    """Forward prop"""
    L = x
    for i in range(len(layer_sizes)):
        L = create_layer(L, layer_sizes[i], activation[i])

    return L
