#!/usr/bin/env python3
"""Eight activations, split over two figures so the panels stay large."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY
import matplotlib.pyplot as plt
import numpy as np
from common.models import ACTIVATIONS

NAME = "activation_zoo"


def _panel(which, name, cols):
    style.use()
    a = np.linspace(-3, 3, 900)
    fig, ax = plt.subplots(1, 4, figsize=(14.0, 4.9))
    for axis, (nm, formula, f, _), c in zip(ax.ravel(), which, cols):
        axis.plot(a, f(a), color=c, lw=3.0)
        axis.axhline(0, color=GREY, lw=0.6, ls=":")
        axis.axvline(0, color=GREY, lw=0.6, ls=":")
        axis.set_title(nm + "\n" + formula, fontsize=15)
        axis.set_xlabel(r"$a$")
        style.square(axis)
    ax[0].set_ylabel(r"$h(a)$")
    fig.tight_layout(w_pad=1.6)
    return style.save(fig, name)


def build():
    _panel(ACTIVATIONS[:4], "activation_zoo_a", [OUTC, OUTC, OUTC, ACC])
    return _panel(ACTIVATIONS[4:], "activation_zoo_b", [ACC, ACC, L2, L2])


if __name__ == "__main__":
    build()
