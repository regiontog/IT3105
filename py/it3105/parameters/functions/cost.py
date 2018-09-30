from .decorators import param_schema


@param_schema(None)
def mean_square_error():
    def inner():
        pass

    return inner


COSTS = [
    mean_square_error
]
