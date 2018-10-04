import copy

from cerberus import Validator

from .parameter import Parameter, Parameters
from .validation import SCHEMA, TRANSFORMATIONS


class NetworkParameters(Parameters):
    def __init__(self, params):
        validator = Validator(SCHEMA)
        assert validator.validate(
            params), "Invalid parameters: {}".format(validator.errors)

        transformed = NetworkParameters.transform(
            validator.document, TRANSFORMATIONS)

        super().__init__(NetworkParameters.recursively_dict_to_parameters(transformed))

    @staticmethod
    def transform(params, transformations):
        results = copy.copy(params)

        for key, transform in transformations.items():
            if isinstance(transform, dict):
                results[key] = NetworkParameters.transform(
                    results[key], transform)
            else:
                transform, nested = transform
                if len(nested) > 0:
                    results[key] = NetworkParameters.transform(
                        results[key], nested)

                results[key] = transform(results[key])

        return results

    @staticmethod
    def recursively_dict_to_parameters(d, schema=SCHEMA):
        result = {}

        for name, value in d.items():
            if name in schema \
                    and 'type' in schema[name] \
                    and schema[name]['type'] == 'dict':
                value = NetworkParameters.recursively_dict_to_parameters(
                    value, schema=schema[name]['schema']
                )

            if isinstance(value, dict) and all('schema' in schema[name] and key in schema[name]['schema'] for key in value):
                value = Parameters(value)

            result[name] = value

        return result

    @staticmethod
    def from_toml(filename=None, raw=None):
        import toml

        if raw is not None:
            return NetworkParameters(toml.loads(raw))
        else:
            return NetworkParameters(toml.load(filename))
