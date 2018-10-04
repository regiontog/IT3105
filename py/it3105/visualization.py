from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QLineEdit, QGridLayout, QLabel, QApplication)
from PyQt5.QtCore import pyqtSignal, QObject, QThread

import pyqtgraph as pg
import scipy.cluster.hierarchy as sch
import random
import time


class Signal(QObject):
    signal = pyqtSignal(int, object)

    def __init__(self, fn):
        super().__init__()

        self.signal.connect(self.on_signal)
        self.fn = fn

    def on_signal(self,  *args, **kwargs):
        self.fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.signal.emit(*args, **kwargs)


class Runner(QThread):
    signal = pyqtSignal()

    def __init__(self, target):
        super().__init__()

        self.target = target

    def run(self):
        try:
            self.target()
        except:
            import traceback
            print(traceback.format_exc())

        self.signal.emit()


def start_ui_and_run(runner):
    import sys
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication([])
    w = QMainWindow()

    cw = pg.GraphicsLayoutWidget()
    w.show()
    w.resize(1920, 1080)
    w.setCentralWidget(cw)

    execution = runner(cw)
    next(execution)

    def run_next():
        try:
            next(execution)
        except StopIteration:
            pass

    def quit():
        sys.exit(app.exit())

    r = Runner(run_next)

    r.signal.connect(quit)
    r.start()

    sys.exit(app.exec_())

#ef dendrogram(features,labels,metric='euclidean',mode='average',ax=None,title='Dendrogram',orient='top',lrot=90.0):
#    ax = ax if ax else PLT.gca()

#    SCH.dendrogram(cluster_history,labels=labels,orientation=orient,leaf_rotation=lrot)
#    PLT.tight_layout()
#    ax.set_title(title)
#    ax.set_ylabel(metric + ' distance')
#    PLT.show()


import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np

#activations = np.random.rand(10, 10)

#cluster_history = sch.linkage(activations, method='average', metric='euclidean')

def dendrogram(features,labels,metric='euclidean',mode='average',ax=None,title='Dendrogram',orient='top',lrot=90.0):
    ax = ax if ax else plt.gca()
    cluster_history = sch.linkage(features,method=mode,metric=metric)
    sch.dendrogram(cluster_history,labels=labels,orientation=orient,leaf_rotation=lrot)
    plt.tight_layout()
    ax.set_title(title)
    ax.set_ylabel(metric + ' distance')
    plt.show()

#dendrogram(activations, ['A', 'B', 'C','A', 'B', 'C','A', 'B', 'C','D'])