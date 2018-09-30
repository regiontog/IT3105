from .decorators import infer_schema


@infer_schema
def from_csv(filepath: str):
    def inner():
        pass

    return inner


SOURCE_GENERATORS = [
    from_csv
]
