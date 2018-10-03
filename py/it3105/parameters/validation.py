from .parameter import Parameter
from .functions.activation import ACTIVATIONS
from .functions.optimize import OPTIMIZERS
from .functions.data import SOURCE_GENERATORS
from .functions.learning_rate import ADAPTIVE_RATES
from .functions.cost import COSTS

from .cerberus import *


SCHEMA = Parameter.normalize({
    Parameter.DIMENSIONS: record({
        Parameter.LAYERS: record({
            Parameter.NUM: integer(),
            Parameter.SIZE: integer()
        }),
        Parameter.INPUT: record({
            Parameter.SIZE: integer()
        }),
        Parameter.OUTPUT: record({
            Parameter.SIZE: integer()
        }),
    }),
    Parameter.ACTIVATION: record({
        Parameter.LAYERS: function(ACTIVATIONS),
        Parameter.OUTPUT: function(ACTIVATIONS)
    }),
    Parameter.COST: function(COSTS),
    Parameter.OPTIMIZER: function(OPTIMIZERS),
    Parameter.INITIAL_WEIGHTS: record({
        Parameter.LOWER: decimal(),
        Parameter.UPPER: decimal(),
    }),
    Parameter.DATA: record({
        Parameter.CASE_FRACTION: decimal(),
        Parameter.SOURCE: function(SOURCE_GENERATORS),
    }),
    Parameter.VALIDATION: record({
        Parameter.FRACTION: decimal(),
        Parameter.INTERVAL: integer(),
    }),
    Parameter.TEST: record({
        Parameter.FRACTION: decimal(),
    }),
    Parameter.MINIBATCH: record({
        Parameter.SIZE: integer(),
        Parameter.STEPS: integer(),
    }),
    Parameter.VISUALIZATION: record({
        Parameter.WEIGHTS: array(integer()),
        Parameter.BIASSES: array(integer()),
        Parameter.MAP: record({
            Parameter.SIZE: integer(),
            Parameter.LAYERS: array(integer()),
            Parameter.DENDOGRAM_LAYERS: array(integer()),
        }),
    })
})


def fn_transform(functions):
    fn_lookup = {fn.__name__: fn for fn in functions}

    def transform(doc):
        if isinstance(doc, dict) and Parameter.get_name(Parameter.FUNCTION) in doc:
            generator_fn = fn_lookup[doc[Parameter.get_name(
                Parameter.FUNCTION)]]

            if Parameter.get_name(Parameter.PARAMETERS) in doc:
                return generator_fn(**doc[Parameter.get_name(Parameter.PARAMETERS)])
            else:
                return generator_fn()
        else:
            return doc

    return transform


TRANSFORMATIONS = Parameter.normalize({
    Parameter.ACTIVATION: Parameter.normalize({
        Parameter.LAYERS: (fn_transform(ACTIVATIONS), {}),
        Parameter.OUTPUT: (fn_transform(ACTIVATIONS), {})
    }),
    Parameter.COST: (fn_transform(COSTS), {}),
    Parameter.OPTIMIZER: (fn_transform(OPTIMIZERS), Parameter.normalize({
        Parameter.PARAMETERS: Parameter.normalize({
            Parameter.LEARNING_RATE: (fn_transform(ADAPTIVE_RATES), {}),
        }),
    })),
    Parameter.DATA: Parameter.normalize({
        Parameter.SOURCE: (fn_transform(SOURCE_GENERATORS), {}),
    }),
})
