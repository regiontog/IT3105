from .decorators import infer_schema


@infer_schema
def RMSProp(learning_rate: float):
    def inner():
        pass

    return inner


OPTIMIZERS = [
    RMSProp
]
