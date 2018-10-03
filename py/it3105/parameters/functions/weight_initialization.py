from .decorators import infer_schema

import tensorflow as tf


@infer_schema
def random_uniform(low: float, high: float):
    return lambda *size: tf.random_uniform(size, minval=low, maxval=high, dtype=tf.float64)


@infer_schema
def truncated_normal(stddev: float):
    return lambda *size: tf.truncated_normal(size, stddev=stddev, dtype=tf.float64)


INITS = [
    random_uniform,
    truncated_normal,
]
