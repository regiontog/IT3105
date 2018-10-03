from .decorators import infer_schema

import tensorflow as tf
import math


@infer_schema
def exponential_decay(
        initial_rate: float,
        decay_steps: float,
        to: float,
        staircase: bool):

    return lambda gs: to + tf.train.exponential_decay(
        initial_rate,
        gs,
        decay_steps,
        1/math.e,
        staircase=staircase,
        name="learning_rate"
    )


ADAPTIVE_RATES = [
    exponential_decay
]
