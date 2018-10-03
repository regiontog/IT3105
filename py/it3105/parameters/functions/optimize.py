from .decorators import infer_schema
from ..cerberus import one_of, decimal, function
from .learning_rate import ADAPTIVE_RATES

import tensorflow as tf

learning_rate_schema = one_of([decimal(), function(ADAPTIVE_RATES)])


def with_std_learning_rate(learning_rate, cls, **kkwargs):
    def inner(global_step, *args, **kwargs):
        if callable(learning_rate):
            lr = learning_rate(global_step)
        else:
            lr = learning_rate
            tf.constant(lr, name="learning_rate")

        opt = cls(learning_rate=lr, **kkwargs)
        return opt.minimize(*args, **kwargs)

    return inner


@infer_schema
def RMSProp(learning_rate: learning_rate_schema):
    return with_std_learning_rate(learning_rate, tf.train.RMSPropOptimizer)


@infer_schema
def gradient_descent(learning_rate: learning_rate_schema):
    return with_std_learning_rate(learning_rate, tf.train.GradientDescentOptimizer)


@infer_schema
def Adam(learning_rate: learning_rate_schema, epsilon: float):
    return with_std_learning_rate(learning_rate, tf.train.AdamOptimizer, epsilon=epsilon)


OPTIMIZERS = [
    RMSProp,
    gradient_descent,
    Adam,
]
