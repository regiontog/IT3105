import functools
import typing


def param_schema(schema):
    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)

        inner._parameters_schema = schema
        return inner

    return decorator


to_schema = {
    str: {
        'type': 'string',
        'required': True,
    },
    float: {
        'type': 'float',
        'required': True,
    }
}


def infer_schema(fn):
    hints = typing.get_type_hints(fn)
    schema = {
        'type': 'dict',
        'required': True,
        'schema': {field_name: to_schema[cls] for field_name, cls in hints.items()},
    }
    return param_schema(schema)(fn)
