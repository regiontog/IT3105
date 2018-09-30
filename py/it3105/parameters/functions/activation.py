from .decorators import param_schema


@param_schema(None)
def softmax():
    def inner():
        pass

    return inner


ACTIVATIONS = [
    softmax
]
