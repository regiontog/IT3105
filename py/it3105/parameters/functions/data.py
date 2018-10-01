from .decorators import infer_schema


@infer_schema
def from_csv(filepath: str):
    def inner():
        pass

    return inner


@infer_schema
def import_dataset(filepath: str, dataset: str):
    import runpy
    from os import path

    def inner(basedir):
        ns = runpy.run_path(path.join(basedir, filepath))
        return ns[dataset]

    return inner


SOURCE_GENERATORS = [
    from_csv,
    import_dataset,
]
