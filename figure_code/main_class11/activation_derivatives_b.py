"""activation_derivatives_b — The same eight differentiated -- what backpropagation multiplies by.

Used in: Class 11 (B), slide(s): "Differentiated: smooth and saturating"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python activation_derivatives_b.py            -> output/activation_derivatives_b.pdf

Each panel marks the maximum of h', because that value is the best case
factor a layer can contribute.  Anything below 1 shrinks the gradient at
every layer; anything above 1 can grow it.

This script draws the last four (GELU, SiLU, tanh, logistic); the original
module wrote _a and _b from one build().
Ported from B_Recreated_Figures/src/figures/fig02_activation_derivatives.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as style
from common import L1, L2, OUTC, ACC, GREY
import matplotlib.pyplot as plt
import numpy as np
from common import ACTIVATIONS

NAME = "activation_derivatives_b"


def _panel(which, name, cols):
    style.use()
    a = np.linspace(-4, 4, 2400)
    fig, ax = plt.subplots(1, 4, figsize=(14.0, 4.9))
    for axis, (nm, _, _, df), c in zip(ax.ravel(), which, cols):
        d = df(a)
        axis.plot(a, d, color=c, lw=3.0)
        axis.axhline(1.0, color=GREY, lw=1.2, ls="--")
        axis.axhline(0, color=GREY, lw=0.6, ls=":")
        m = float(np.max(d)); am = float(a[int(np.argmax(d))])
        # mark the maximum
        axis.plot([am], [m], marker="o", ms=11, color=L1, mec="white",
                  mew=1.4, zorder=6)
        axis.annotate(rf"$\max h'={m:.2f}$", xy=(am, m),
                      xytext=(am + 0.35, m + 0.16), color=L1, fontsize=13,
                      ha="left")
        axis.set_title(nm, fontsize=15)
        axis.set(xlabel=r"$a$", ylim=(-0.2, 1.6))
        if m < 0.6:
            axis.axhspan(-0.2, 1.6, color=OUTC, alpha=.06, lw=0)
        style.square(axis)
    ax[0].set_ylabel(r"$h'(a)$")
    fig.tight_layout(w_pad=1.6)
    return style.save(fig, name)


def build():
    return _panel(ACTIVATIONS[4:], NAME,
                  [ACC, ACC, L2, L2])


if __name__ == "__main__":
    build()
