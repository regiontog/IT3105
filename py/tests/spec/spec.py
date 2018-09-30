from it3105.parameters import NetworkParameters, Parameter as P

toml_str = """
[dimensions.layers]
num = 4
size = 5

[dimensions.input]
size = 10

[activation.layers]
function = "softmax"

[activation.output]
function = "softmax"

[cost]
function = "mean_square_error"

[optimizer]
function = "RMSProp"

[optimizer.parameters]
learning_rate = 0.30

[initial_weights]
lower = 0.0
upper = 1.0

[data]
case_fraction = 1.0

[data.source]
function = "from_csv"

[data.source.parameters]
filepath = "/home/alan/ann/data/wine.csv"

[validation]
fraction = 0.5
interval = 100

[test]
fraction = 0.3

[minibatch]
size = 100
steps = 1000

[visualization]
weights = [1, 2, 3, 4]
biasses = [1, 4]

[visualization.map]
size = 0
layers = [0, 5]
dendogram_layers = [5]
"""


def test_spec():
    spec = NetworkParameters.from_toml(raw=toml_str)

    assert spec[P.DIMENSIONS][P.LAYERS][P.SIZE] == 5
    assert spec[P.DIMENSIONS][P.LAYERS][P.NUM] == 4
    assert spec[P.DIMENSIONS][P.INPUT][P.SIZE] == 10
    assert callable(spec[P.ACTIVATION][P.LAYERS])
    assert callable(spec[P.ACTIVATION][P.OUTPUT])
    assert callable(spec[P.COST])
