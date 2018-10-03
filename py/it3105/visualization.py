from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QLineEdit, QGridLayout, QLabel, QApplication)
from PyQt5.QtCore import pyqtSignal, QObject, QThread

import pyqtgraph as pg
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
    def __init__(self, target):
        super().__init__()

        self.target = target

    def run(self):
        self.target()


def start_ui_and_run(runner):
    import sys

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

    r = Runner(run_next)
    r.start()

    sys.exit(app.exec_())
