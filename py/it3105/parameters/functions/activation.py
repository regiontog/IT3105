from .decorators import param_schema

import tensorflow as tf


@param_schema(None)
def softmax():
    return tf.nn.softmax


@param_schema(None)
def relu():
    return tf.nn.relu


@param_schema(None)
def sigmoid():
    return tf.nn.sigmoid


ACTIVATIONS = [
    softmax,
    relu,
    sigmoid,
]
