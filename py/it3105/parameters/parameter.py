from enum import auto, Enum


class Parameter(Enum):
    MAP = auto()
    NUM = auto()
    SIZE = auto()
    COST = auto()
    DATA = auto()
    TEST = auto()
    LOWER = auto()
    UPPER = auto()
    INPUT = auto()
    STEPS = auto()
    SOURCE = auto()
    LAYERS = auto()
    OUTPUT = auto()
    WEIGHTS = auto()
    BIASSES = auto()
    FUNCTION = auto()
    FRACTION = auto()
    INTERVAL = auto()
    OPTIMIZER = auto()
    MINIBATCH = auto()
    DIMENSIONS = auto()
    ACTIVATION = auto()
    PARAMETERS = auto()
    VALIDATION = auto()
    VISUALIZATION = auto()
    CASE_FRACTION = auto()
    INITIAL_WEIGHTS = auto()
    DENDOGRAM_LAYERS = auto()

    @staticmethod
    def normalize(dict):
        return {Parameter.get_name(param): value for param, value in dict.items()}

    @staticmethod
    def get_name(param):
        return param.name.lower()


class Parameters:
    def __init__(self, params):
        self.parameters = params

    def __getitem__(self, param):
        return self.parameters[Parameter.get_name(param)]
