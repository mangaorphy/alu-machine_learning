#!//usr/bin/env python3
"""Create layer"""
import tensorflow as tf

def create_layer(prev, n, activation):
    """Create layer"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    L = tf.layers.Dense(units=n, activation=activation, kernel_initializer=W, name="layer")
    return L(prev)
