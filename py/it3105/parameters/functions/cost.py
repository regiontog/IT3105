from .decorators import param_schema

import tensorflow as tf


@param_schema(None)
def mean_square_error():
    return lambda batch_size, y_log, y, y_: tf.losses.mean_squared_error(y, y_)


@param_schema(None)
def softmax_cross_entropy():
    def ce(batch_size, y_log, y, y_):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=y_log, labels=y))*batch_size

    return ce


@param_schema(None)
def sigmoid_cross_entropy():
    def ce(batch_size, y_log, y, y_):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=y_log, labels=y))*batch_size

    return ce


COSTS = [
    mean_square_error,
    sigmoid_cross_entropy,
    softmax_cross_entropy,
]


@param_schema(None)
def argmax():
    def am(y, y_):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

    return am


ACCURACY_TESTS = [
    argmax
]
