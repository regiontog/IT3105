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

import pyqtgraph as pg

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


def draw_rect(vbox, x, y, width, height, pen=pg.mkBrush('r')):
    r = pg.QtGui.QGraphicsRectItem(x, y, width, height)
    r.setPen(pg.mkPen(None))
    r.setBrush(pen)
    vbox.addItem(r)


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

    dendogram_layers = spec[P.VISUALIZATION][P.DENDOGRAM_LAYERS]

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

    grab_vars = []
    activations = []

    if 0 in dendogram_layers:
        activations.append(minibatch_input)    

    for i, layer in enumerate(spec[P.LAYERS]):
        #if i in dendogram_layers:
        #    activations.append(logits)
        
        layer_gen = LAYER_TYPES[layer[P.get_name(P.KIND)]]
        activation_fn, logits, prev_layer_size, grab = layer_gen(
            layer, init_w, train_feed, test_feed)(prev_layer_size, prev_layer)

        if grab:
            grab_vars.append(grab)

        prev_layer = activation_fn(logits)

        if i in dendogram_layers:
            activations.append(logits)

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

        # Training
        for epoch, (xs, ys) in zip(range(mb_steps), minibatches):
            testr, outr, lr, lossr, _, _ = session.run([test, output, learning_rate, loss, train, inc_global_step], {
                **train_feed,
                minibatch_input: xs,
                label: ys,
            })

            plot_error(epoch, lossr)
            plot_lr(epoch, lr)
            plot_tt(epoch, testr)

            # Validation
            if epoch % vint == 0:
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

        import code

        def y(x):
            return session.run(output, {
                **test_feed,
                minibatch_input: [x],
            })[0]

        # Mapping
        map_size = spec[P.VISUALIZATION][P.SIZE]

        if map_size > 0:
            xs, ys = zip(*islice(dataset.stream_cases(), map_size))

            grabr, acsr = session.run([grab_vars, activations], {
                **test_feed,
                minibatch_input: xs,
                label: ys
            })

            post_training_analysis(cw, grabr, acsr)

            code.InteractiveConsole(locals=dict(
                y=y,
                grabbed_vars=grabr,
                activations=acsr,
            )).interact()
        else:
            code.InteractiveConsole(locals=dict(
                y=y,
            )).interact()


def post_training_analysis(cw, grabbed_vars, activations):
    # mapping_box = cw.addViewBox(row=1, col=0)

    # draw_rect(mapping_box, 0, 0, 0.2, 0.2, pen=pg.mkBrush('g'))
    # draw_rect(mapping_box, 0, 0, 0.1, 0.1)

    # Dendograms
    for activation in activations:
        labels = list(map(str, range(len(activation[0]))))

        if len(labels) > 1:
            dendrogram(np.transpose(activation), labels)

    # Mapping
    
    for activation in activations:
        hinton_plot(activation)

    
    """
    for grabbed_var in grabbed_vars:
        for grab in grabbed_var:
            hinton_plot(grab)
    """

# TODO: graphing oraganize and use pyqtgraph
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def dendrogram(features, labels, metric='euclidean', mode='average', ax=None, title='Dendrogram', orient='top', lrot=90.0):
    ax = ax if ax else plt.gca()
    cluster_history = sch.linkage(features, method=mode, metric=metric)
    sch.dendrogram(cluster_history, labels=labels,
                   orientation=orient, leaf_rotation=lrot)
    plt.tight_layout()
    ax.set_title(title)
    ax.set_ylabel(metric + ' distance')
    plt.show()


# This is Hinton's classic plot of a matrix (which may represent snapshots of weights or a time series of
# activation values).  Each value is represented by a red (positive) or blue (negative) square whose size reflects
# the absolute value.  This works best when maxsize is hardwired to 1.  The transpose (trans) arg defaults to
# true so that matrices are plotted with rows along a horizontal plane, with the 0th row on top.

# The 'colors' argument, a list, is ordered as follows: background, positive-value, negative-value, box-edge.
# If you do not want to draw box edges, just use 'None' as the 4th color.  A gray-scale combination that
# mirrors Hinton's original version is ['gray','white','black',None]
def hinton_plot(matrix, maxval=None, maxsize=1, fig=None, trans=True, scale=True, title='Hinton plot',
                colors=['gray', 'red', 'blue', 'white']):
    hfig = fig if fig else plt.figure()
    hfig.suptitle(title, fontsize=18)
    if trans:
        matrix = matrix.transpose()
    if maxval == None:
        maxval = np.abs(matrix).max()
    if not maxsize:
        maxsize = 2**np.ceil(np.log(maxval)/np.log(2))

    axes = hfig.gca()
    axes.clear()
    # This is the background color.  Hinton uses gray
    axes.patch.set_facecolor(colors[0])
    # Options: ('equal'), ('equal','box'), ('auto'), ('auto','box')..see matplotlib docs
    axes.set_aspect('auto', 'box')
    axes.xaxis.set_major_locator(plt.NullLocator())
    axes.yaxis.set_major_locator(plt.NullLocator())

    ymax = (matrix.shape[1] - 1) * maxsize
    for (x, y), val in np.ndenumerate(matrix):
        # Hinton uses white = pos, black = neg
        color = colors[1] if val > 0 else colors[2]
        if scale:
            size = max(0.01, np.sqrt(min(maxsize, maxsize*np.abs(val)/maxval)))
        else:
            # The original version did not include scaling
            size = np.sqrt(min(np.abs(val), maxsize))
        # (ymax - y) to invert: row 0 at TOP of diagram
        bottom_left = [x - size / 2, (ymax - y) - size / 2]
        blob = plt.Rectangle(bottom_left, size, size,
                             facecolor=color, edgecolor=colors[3])
        axes.add_patch(blob)
    axes.autoscale_view()
    plt.show()


def ensure(this, err_msg, warning=False):
    if not this:
        print(err_msg)
        if not warning:
            exit(-1)


def chunks(iterable, chunksize):
    args = [iter(iterable)] * chunksize

    return zip_longest(*args)
