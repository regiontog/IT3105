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
    ploss.addLegend()
    plot_error = curve_plot_signal(ploss, name="Training", pen='g')
    plot_vl = curve_plot_signal(ploss, name="Validation", pen='w')

    plr = cw.addPlot(title="Learning rate")
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
    input_size = spec[P.DIMENSIONS][P.INPUT][P.SIZE]
    output_size = spec[P.DIMENSIONS][P.OUTPUT][P.SIZE]
    mb_size = spec[P.MINIBATCH][P.SIZE]
    hl_num = spec[P.DIMENSIONS][P.LAYERS][P.NUM]
    hl_size = spec[P.DIMENSIONS][P.LAYERS][P.SIZE]

    lactivation = spec[P.ACTIVATION][P.LAYERS]
    oactivation = spec[P.ACTIVATION][P.OUTPUT]

    cost = spec[P.COST]
    optimize = spec[P.OPTIMIZER]

    # Unzipping datasetes
    minibatches = (zip(*chunk) for chunk in chunks(training, mb_size))
    validation = list(zip(*validation))
    testing = list(zip(*testing))

    # Initialize the layer's variables if they should not be loaded
    low = spec[P.INITIAL_WEIGHTS][P.LOWER]
    high = spec[P.INITIAL_WEIGHTS][P.UPPER]

    def initial_w(*size):
        return np.random.uniform(low, high, size=size)

    # Should use w = U([0,n]) * sqrt(2.0/n)???
    layers = [(tf.Variable(initial_w(hl_size, hl_size), dtype=tf.float64), tf.Variable(np.zeros(hl_size), dtype=tf.float64))
              for _ in range(hl_num - 1)]

    layers.insert(0, (
        tf.Variable(initial_w(input_size, hl_size), dtype=tf.float64),
        tf.Variable(np.zeros(hl_size), dtype=tf.float64)
    ))

    layers.append((
        tf.Variable(initial_w(hl_size, output_size), dtype=tf.float64),
        tf.Variable(np.zeros(output_size), dtype=tf.float64)
    ))

    minibatch_input = tf.placeholder(tf.float64)
    label = tf.placeholder(tf.float64)

    activations = [minibatch_input]

    for weights, biasses in layers[:-1]:
        activations.append(lactivation(
            activations[-1] @ weights + biasses
        ))

    oweights, obiasses = layers[-1]
    output = oactivation(
        activations[-1] @ oweights + obiasses
    )

    activations.append(output)

    variables = list(chain(*layers))

    global_step = tf.Variable(1, name='global_step',
                              trainable=False, dtype=tf.int32)
    inc_global_step = tf.assign(global_step, global_step+1)

    loss = cost(label, output)
    # train = optimize(global_step, loss, var_list=variables)
    # Ensure we train variables created by the optimizer as well?
    train = optimize(global_step, loss)

    learning_rate = tf.get_default_graph().get_tensor_by_name("learning_rate:0")

    # Run with data
    mb_steps = spec[P.MINIBATCH][P.STEPS]
    vint = spec[P.VALIDATION][P.INTERVAL]

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch, (xs, ys) in zip(range(mb_steps), minibatches):
            # Training
            outr, lr, lossr, _, _ = session.run([output, learning_rate, loss, train, inc_global_step], {
                minibatch_input: xs,
                label: ys,
            })

            plot_error(epoch, lossr)
            plot_lr(epoch, lr)

            # print(outr[0])
            # print(ys[0])

            if epoch % vint == 0:
                # Validation
                xs, ys = validation
                validation_loss = session.run(loss, {
                    minibatch_input: xs,
                    label: ys
                })

                plot_vl(epoch, validation_loss)

        # Testing
        xs, ys = testing
        y_hat = session.run(output, {
            minibatch_input: xs,
        })

        # for yp in y_hat[:10]:
        #     print("{} => {}".format(yp, onehot(yp)))

        # for y, yp in zip(ys[:10], y_hat):
        #     print("{} =?= {}".format(y, onehot(yp)))

        print("Testing: {}%".format(percent(map(test, zip(y_hat, ys)))))

        def y(x):
            return onehot(session.run(output, {
                minibatch_input: [x],
            })[0])

        import code
        code.InteractiveConsole(locals=dict(y=y)).interact()


def percent(iterable):
    l = list(iterable)
    return 100*sum(1 if x else 0 for x in l)/len(l)


def onehot(y):
    onehot = [0] * len(y)
    onehot[np.argmax(y)] = 1

    return onehot


def test(arg):
    y_hat, y = arg

    return y == onehot(y_hat)


def ensure(this, err_msg, warning=False):
    if not this:
        print(err_msg)
        if not warning:
            exit(-1)


def chunks(iterable, chunksize):
    args = [iter(iterable)] * chunksize

    return zip_longest(*args)
