from .decorators import param_schema

import tensorflow as tf


@param_schema(None)
def mean_square_error():
    return tf.losses.mean_squared_error


COSTS = [
    mean_square_error
]
