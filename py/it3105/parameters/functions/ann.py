from .activation import ACTIVATIONS
from .. import Parameter as P
from .decorators import param_schema
from ..cerberus import string, integer, record, decimal, boolean, optional

import tensorflow as tf

activation_map = {a.__name__: a for a in ACTIVATIONS}

@param_schema({
    P.ACTIVATION: string(),
    P.SIZE: integer(),
    P.VISUALIZATION: optional(record({
        P.WEIGHTS: optional(boolean()),
        P.BIASSES: optional(boolean()),
    }))
})
def regular_feedforward(parameters, init_w, training_feed_dict, test_feed_dict):
    activation_fn = activation_map[parameters[P.get_name(P.ACTIVATION)]]()
    size = parameters[P.get_name(P.SIZE)]

    if P.get_name(P.VISUALIZATION) in parameters:
        visualize = parameters[P.get_name(P.VISUALIZATION)]

        visualize_w = visualize.get(P.get_name(P.WEIGHTS), False)
        visualize_b = visualize.get(P.get_name(P.BIASSES), False)
    else:
        visualize_w = False
        visualize_b = False

    def build_layer(prev_size, prev_layer):
        W, b = (
            tf.Variable(init_w(prev_size, size), dtype=tf.float64),
            tf.Variable(tf.zeros(size, dtype=tf.float64), dtype=tf.float64)
        )

        grab = []

        if visualize_w:
            grab.append(W)

        if visualize_b:
            grab.append(b)

        return activation_fn, prev_layer @ W + b, size, grab

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
        return lambda x: x, tf.nn.dropout(prev_layer, pkeep_ph), prev_size, []

    return build_layer

@param_schema(None)
def convolution(parameters, init_w, training_feed_dict, test_feed_dict):
    def build_layer(prev_size, prev_layer):
        return lambda x: x, tf.nn.conv2d(prev_layer, strides=[1, 2, 2, 1], padding='SAME'), prev_size, []
    return build_layer


@param_schema({
    P.ACTIVATION: string()
})  
def max_pool(parameters, init_w, training_feed_dict, test_feed_dict):
    activation_fn = activation_map[parameters[P.get_name(P.ACTIVATION)]]()
    def build_layer(prev_size, prev_layer):
        return activation_fn, tf.nn.max_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), prev_size, []
    return build_layer

LAYER_TYPES = {l.__name__: l for l in [
    regular_feedforward,
    dropout,
    convolution,
    max_pool
]}