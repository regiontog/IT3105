from .parameter import Parameter
from .functions.optimize import OPTIMIZERS
from .functions.data import SOURCE_GENERATORS
from .functions.learning_rate import ADAPTIVE_RATES
from .functions.cost import COSTS, ACCURACY_TESTS
from .functions.ann import LAYER_TYPES
from .functions.weight_initialization import INITS

from .cerberus import *


def to_layer_schema(item):
    name, fn = item
    schema = {}

    if fn._parameters_schema:
        schema = fn._parameters_schema

    return record({
        **schema,
        Parameter.KIND: string(allowed=[name]),
    })


SCHEMA = Parameter.normalize({
    Parameter.LAYERS: array(one_of(map(to_layer_schema, LAYER_TYPES.items()))),
    Parameter.COST: function(COSTS),
    Parameter.INITIAL_WEIGHTS: function(INITS),
    Parameter.INPUT: record({
        Parameter.SIZE: integer()
    }),
    Parameter.OPTIMIZER: function(OPTIMIZERS),
    Parameter.DATA: record({
        Parameter.CASE_FRACTION: decimal(),
        Parameter.SOURCE: function(SOURCE_GENERATORS),
    }),
    Parameter.VALIDATION: record({
        Parameter.FRACTION: decimal(),
        Parameter.INTERVAL: integer(),
    }),
    Parameter.ACCURACY: function(ACCURACY_TESTS),
    Parameter.TEST: record({
        Parameter.FRACTION: decimal(),
    }),
    Parameter.RANDOM_SEED: decimal(),
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
    Parameter.INITIAL_WEIGHTS: (fn_transform(INITS), {}),
    Parameter.COST: (fn_transform(COSTS), {}),
    Parameter.OPTIMIZER: (fn_transform(OPTIMIZERS), Parameter.normalize({
        Parameter.PARAMETERS: Parameter.normalize({
            Parameter.LEARNING_RATE: (fn_transform(ADAPTIVE_RATES), {}),
        }),
    })),
    Parameter.DATA: Parameter.normalize({
        Parameter.SOURCE: (fn_transform(SOURCE_GENERATORS), {}),
    }),
    Parameter.ACCURACY: (fn_transform(ACCURACY_TESTS), {}),
})
