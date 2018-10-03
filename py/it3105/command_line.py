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
from itertools import zip_longest, islice, chain
import numpy as np

from .parameters import NetworkParameters, Parameter as P
from .visualization import start_ui_and_run, Signal

from .parameters.functions.ann import LAYER_TYPES


DEFAULT_PARSER = NetworkParameters.from_toml

PARSER = {
    '.toml': NetworkParameters.from_toml,
}


def run():
    start_ui_and_run(in_thread)


def curve_plot_signal(plot, name=None, pen='w'):
    curve = plot.plot(name=name, pen=pen)
    data = []
    ys = []

    @Signal
    def plotter(y, x):
        data.append(x)
        ys.append(y)
        curve.setData(ys, data)

    return plotter


def in_thread(cw):
    ploss = cw.addPlot(title="Loss")
    ploss.addLegend(offset=(500, 30))
    plot_error = curve_plot_signal(ploss, name="Training", pen='g')
    plot_vl = curve_plot_signal(ploss, name="Validation", pen='w')

    ptest = cw.addPlot(title="Accuracy")
    plot_tt = curve_plot_signal(ptest, name="Training", pen='g')
    plot_vt = curve_plot_signal(ptest, name="Validation", pen='w')

    plr = cw.addPlot(title="Learning rate")
    plr.setYRange(0, 0.1)
    plr.enableAutoRange()
    plot_lr = curve_plot_signal(plr)

    # Run rest in thread seperated from the GUI thread,
    # all plotting must be done with thread synchronization from here on
    yield

    import tensorflow as tf

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
    testf = spec[P.TEST][P.FRACTION]

    training, validation, testing = dataset.split(
        cf, vf, testf
    )

    # Building network
    input_size = spec[P.INPUT][P.SIZE]
    mb_size = spec[P.MINIBATCH][P.SIZE]

    cost = spec[P.COST]
    optimize = spec[P.OPTIMIZER]
    accuracy = spec[P.ACCURACY]

    tf.set_random_seed(spec[P.RANDOM_SEED])

    # Unzipping datasetes
    minibatches = (zip(*chunk) for chunk in chunks(training, mb_size))
    validation = list(zip(*validation))
    testing = list(zip(*testing))

    # Initialize the layer's variables if they should not be loaded
    init_w = spec[P.INITIAL_WEIGHTS]

    minibatch_input = tf.placeholder(tf.float64)
    label = tf.placeholder(tf.float64)

    prev_layer_size = input_size
    prev_layer = minibatch_input

    train_feed = {}
    test_feed = {}

    for layer in spec[P.LAYERS]:
        layer_gen = LAYER_TYPES[layer[P.get_name(P.KIND)]]
        activation_fn, logits, prev_layer_size = layer_gen(
            layer, init_w, train_feed, test_feed)(prev_layer_size, prev_layer)
        prev_layer = activation_fn(logits)

    output = prev_layer

    global_step = tf.Variable(1, name='global_step',
                              trainable=False, dtype=tf.int32)
    inc_global_step = tf.assign(global_step, global_step+1)

    test = accuracy(label, output)
    loss = cost(mb_size, logits, label, output)
    train = optimize(global_step, loss)

    learning_rate = tf.get_default_graph().get_tensor_by_name("learning_rate:0")

    # Run with data
    mb_steps = spec[P.MINIBATCH][P.STEPS]
    vint = spec[P.VALIDATION][P.INTERVAL]

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch, (xs, ys) in zip(range(mb_steps), minibatches):

            # Training
            testr, lr, lossr, _, _ = session.run([test, learning_rate, loss, train, inc_global_step], {
                **train_feed,
                minibatch_input: xs,
                label: ys,
            })

            plot_error(epoch, lossr)
            plot_lr(epoch, lr)
            plot_tt(epoch, testr)

            if epoch % vint == 0:
                # Validation
                xs, ys = validation
                testr, validation_loss = session.run([test, loss], {
                    **test_feed,
                    minibatch_input: xs,
                    label: ys
                })

                plot_vl(epoch, validation_loss)
                plot_vt(epoch, testr)

        # Testing
        xs, ys = testing
        testr = session.run(test, {
            **test_feed,
            minibatch_input: xs,
            label: ys
        })

        print("Testing: {}%".format(testr * 100))
        print("Error rate: {}%".format((1-testr) * 100))

        def y(x):
            return session.run(output, {
                **test_feed,
                minibatch_input: [x],
            })[0]

        import code
        code.InteractiveConsole(locals=dict(y=y)).interact()
        print("done")


def ensure(this, err_msg, warning=False):
    if not this:
        print(err_msg)
        if not warning:
            exit(-1)


def chunks(iterable, chunksize):
    args = [iter(iterable)] * chunksize

    return zip_longest(*args)
