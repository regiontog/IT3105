from .decorators import infer_schema

import tensorflow as tf


@infer_schema
def exponential_decay(
        initial_rate: float,
        decay_steps: float,
        decay_rate: float,
        staircase: bool):

    return lambda gs: tf.train.exponential_decay(
        initial_rate,
        gs,
        decay_steps,
        decay_rate,
        staircase=staircase,
        name="learning_rate"
    )


ADAPTIVE_RATES = [
    exponential_decay
]
