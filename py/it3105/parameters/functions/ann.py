from .activation import ACTIVATIONS
from .. import Parameter as P
from .decorators import param_schema
from ..cerberus import string, integer, record, decimal

import tensorflow as tf


activation_map = {a.__name__: a for a in ACTIVATIONS}


@param_schema({
    P.ACTIVATION: string(),
    P.SIZE: integer(),
})
def regular_feedforward(parameters, init_w, training_feed_dict, test_feed_dict):
    activation_fn = activation_map[parameters[P.get_name(P.ACTIVATION)]]()
    size = parameters[P.get_name(P.SIZE)]

    def build_layer(prev_size, prev_layer):
        W, b = (
            tf.Variable(init_w(prev_size, size), dtype=tf.float64),
            tf.Variable(tf.zeros(size, dtype=tf.float64), dtype=tf.float64)
        )

        return activation_fn, prev_layer @ W + b, size

    return build_layer


@param_schema({
    P.PKEEP: decimal(),
})
def dropout(parameters, init_w, training_feed_dict, test_feed_dict):
    pkeep = parameters[P.get_name(P.PKEEP)]
    pkeep_ph = tf.placeholder(tf.float64)

    training_feed_dict[pkeep_ph] = pkeep
    test_feed_dict[pkeep_ph] = 1

    def build_layer(prev_size, prev_layer):
        return lambda x: x, tf.nn.dropout(prev_layer, pkeep_ph), prev_size

    return build_layer


LAYER_TYPES = {l.__name__: l for l in [
    regular_feedforward,
    dropout,
]}
