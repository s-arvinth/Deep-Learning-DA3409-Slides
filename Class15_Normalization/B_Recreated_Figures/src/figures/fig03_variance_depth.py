"""What the initialization variance does over fifty layers.

A network of 50 layers with 100 units each, weights drawn from
N(0, sigma^2) and ReLU activations. One layer multiplies the variance
by D sigma^2 / 2, so over K layers the factor is that number to the Kth
power: geometric growth or geometric decay, with a single value of
sigma^2 on the knife edge.

(a) the variance of the pre-activations going forward.
(b) the variance of the gradient signal coming back.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import forward_variances, backward_variances

NAME = "variance_depth"

D, K = 100, 50
SIGMAS = [0.001, 0.01, 0.02, 0.1, 1.0]


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    cols = plt.get_cmap(S.CAT)(np.linspace(0.05, 0.80, len(SIGMAS)))
    layers = np.arange(K + 1)

    for ax, fn, title, ylab in (
            (axes[0], forward_variances, "(a) forward pass",
             "variance of the pre-activations"),
            (axes[1], backward_variances, "(b) backward pass",
             "variance of the gradient")):
        for s2, col in zip(SIGMAS, cols):
            v = fn(D=D, K=K, sigma2=s2, seed=2)
            v = np.clip(v, 1e-30, 1e30)
            lab = r"$\sigma^2_w = %g$" % s2
            if abs(s2 - 2.0 / D) < 1e-9:
                ax.semilogy(layers, v, color=S.OUTC, lw=3.0, zorder=6,
                            label=lab + r"  $= 2/D$")
            else:
                ax.semilogy(layers, v, color=col, lw=2.1, label=lab)
        ax.set_xlabel("layer")
        ax.set_ylabel(ylab)
        ax.set_ylim(1e-26, 1e26)
        ax.set_yticks([1e-24, 1e-12, 1e0, 1e12, 1e24])
        ax.legend(loc="upper left", fontsize=11.5)
        ax.set_title(title)
        S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
