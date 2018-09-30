from .parameter import Parameter
from .functions.activation import ACTIVATIONS
from .functions.optimize import OPTIMIZERS
from .functions.data import SOURCE_GENERATORS
from .functions.cost import COSTS


def record(schema, **kwargs):
    return {
        **kwargs,
        'type': 'dict',
        'schema': Parameter.normalize(schema),
        'required': True,
    }


def integer(**kwargs):
    return {
        **kwargs,
        'type': 'integer',
        'required': True,
    }


def decimal(**kwargs):
    return {
        **kwargs,
        'type': 'float',
        'required': True,
    }


def string(**kwargs):
    return {
        **kwargs,
        'type': 'string',
        'required': True,
    }


def one_of(choices):
    return {
        'oneof': list(choices),
        'required': True,
    }


def array(T, **kwargs):
    return {
        **kwargs,
        'type': 'list',
        'schema': T,
        'required': True,
    }


def function(functions, **kwargs):
    def to_fn(function):
        schema = {
            **kwargs,
            Parameter.FUNCTION: string(allowed=[function.__name__]),
        }

        if function._parameters_schema:
            schema[Parameter.PARAMETERS] = function._parameters_schema

        return record(schema)

    return one_of(map(to_fn, functions))


SCHEMA = Parameter.normalize({
    Parameter.DIMENSIONS: record({
        Parameter.LAYERS: record({
            Parameter.NUM: integer(),
            Parameter.SIZE: integer()
        }),
        Parameter.INPUT: record({
            Parameter.SIZE: integer()
        })
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
        generator_fn = fn_lookup[doc[Parameter.get_name(Parameter.FUNCTION)]]

        if Parameter.get_name(Parameter.PARAMETERS) in doc:
            return generator_fn(**doc[Parameter.get_name(Parameter.PARAMETERS)])
        else:
            return generator_fn()

    return transform


TRANSFORMATIONS = Parameter.normalize({
    Parameter.ACTIVATION: Parameter.normalize({
        Parameter.LAYERS: fn_transform(ACTIVATIONS),
        Parameter.OUTPUT: fn_transform(ACTIVATIONS)
    }),
    Parameter.COST: fn_transform(COSTS),
    Parameter.OPTIMIZER: fn_transform(OPTIMIZERS),
    Parameter.DATA: Parameter.normalize({
        Parameter.SOURCE: fn_transform(SOURCE_GENERATORS),
    }),
})
