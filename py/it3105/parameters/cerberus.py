from .parameter import Parameter


def record(schema, **kwargs):
    return {
        **kwargs,
        'type': 'dict',
        'schema': Parameter.normalize(schema),
        'required': True,
    }


def optional(schema):
    return {
        **schema,
        'required': False,
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


def boolean(**kwargs):
    return {
        **kwargs,
        'type': 'boolean',
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


def shape(**kwargs):
    return array(integer(**kwargs))


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
