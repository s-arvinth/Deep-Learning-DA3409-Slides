"""activations — Six activation functions on common axes

Used in: Class 5 (B), slide(s): "Other activation functions"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python activations.py            -> output/activations.pdf
Ported from: Class05_Shallow_NN_Regression/B_Recreated_Figures/src/figures/fig06_activations.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import L2, OUTC, ACC, GREY, relu

NAME = "activations"


def build():
    common.use()
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
        axis.set_title(f"{nm}\n{formula}", fontsize=15.6)
        axis.set_xlabel(r"$a$")
        common.square(axis)
    for r in range(2):
        ax[r, 0].set_ylabel(r"$h(a)$")
    fig.tight_layout(w_pad=1.8, h_pad=2.2)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
