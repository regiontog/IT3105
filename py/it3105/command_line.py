"""Usage: spec [-vq] FILE
          spec (-h | --help)

Create and run neural network specified by FILE.

Arguments:
  FILE        input file in one of the supported file types (toml)

Options:
  -h --help
  -v       verbose mode
  -q       quiet mode
"""
from .parameters import NetworkParameters, Parameter as P


DEFAULT_PARSER = NetworkParameters.from_toml

PARSER = {
    '.toml': NetworkParameters.from_toml,
}


def run():
    from os import path
    from docopt import docopt

    from .abc import Dataset

    arguments = docopt(__doc__)
    filepath = path.abspath(arguments['FILE'])

    basedir = path.dirname(filepath)
    _, ext = path.splitext(filepath)
    ensure(ext in PARSER,
           "Extention {} is not supported. Assuming toml format!".format(ext), warning=True)

    spec = PARSER.get(ext, DEFAULT_PARSER)(filepath)

    dataset: Dataset = spec[P.DATA][P.SOURCE](basedir)

    cf = spec[P.DATA][P.CASE_FRACTION]
    vf = spec[P.VALIDATION][P.FRACTION]
    tf = spec[P.TEST][P.FRACTION]

    training, validation, testing = dataset.split(
        cf, vf, tf
    )

    print("Validation")
    print(validation)

    print("Testing")
    print(testing)

    print("Training")
    print(training)


def ensure(this, err_msg, warning=False):
    if not this:
        print(err_msg)
        if not warning:
            exit(-1)
