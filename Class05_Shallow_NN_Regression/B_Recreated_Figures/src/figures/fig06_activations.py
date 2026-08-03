#!/usr/bin/env python3
"""Six activation functions on common axes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import relu

NAME = "activations"


def build():
    style.use()
    a = np.linspace(-3, 3, 700)
    fns = [(r"ReLU", r"$\max(0,a)$", relu(a), OUTC),
           (r"leaky ReLU", r"$\max(0.1a,\,a)$",
            np.where(a > 0, a, 0.1 * a), OUTC),
           (r"tanh", r"$\tanh a$", np.tanh(a), L2),
           (r"logistic sigmoid", r"$1/(1+e^{-a})$",
            1 / (1 + np.exp(-a)), L2),
           (r"softplus", r"$\ln(1+e^{a})$", np.log1p(np.exp(a)), ACC),
           (r"GELU", r"$a\,\Phi(a)$",
            a * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                   (a + 0.044715 * a ** 3))), ACC)]

    fig, ax = plt.subplots(2, 3, figsize=(10.8, 6.6))
    for axis, (nm, formula, ya, c) in zip(ax.ravel(), fns):
        axis.plot(a, ya, color=c, lw=2.4)
        axis.axhline(0, color=GREY, lw=0.5, ls=":")
        axis.axvline(0, color=GREY, lw=0.5, ls=":")
        axis.set_title(f"{nm}\n{formula}", fontsize=12)
        axis.set_xlabel(r"$a$")
        style.square(axis)
    for r in range(2):
        ax[r, 0].set_ylabel(r"$h(a)$")
    fig.tight_layout(w_pad=1.8, h_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
