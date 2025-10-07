#!/usr/bin/env python3
"""calculate loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """loss function"""
    return tf.losses.softmax_cross_entropy(y , y_pred)
