import functools
import typing

from ..cerberus import string, decimal, boolean


def param_schema(schema):
    def decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)

        inner._parameters_schema = schema
        return inner

    return decorator


types = {
    str: string(),
    float: decimal(),
    bool: boolean()
}


def to_schema(cls):
    if isinstance(cls, dict):
        return cls
    else:
        return types[cls]


def infer_schema(fn):
    hints = typing.get_type_hints(fn)
    schema = {
        'type': 'dict',
        'required': True,
        'schema': {field_name: to_schema(cls) for field_name, cls in hints.items()},
    }
    return param_schema(schema)(fn)
