"""What a normalization layer buys during training.

The same eight-layer network, the same data, the same initialization.
(a) training error at three learning rates without normalization: the
largest is unstable. (b) the same three with batch normalization: all
three are stable, and the largest is now the fastest.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (make_net, he_variance, batch_norm,
                           batch_norm_backward, train)

NAME = "bn_effect"

DIMS = [40] + [40] * 6 + [1]
ETAS = [0.01, 0.05, 0.20]


def _data(n=512, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, DIMS[0]))
    Y = np.sin(1.3 * X[:, :1]) + 0.4 * X[:, 1:2]
    return X, Y


def _smooth(x, k=15):
    return np.array([np.median(x[max(0, i - k):i + 1])
                     for i in range(len(x))])


def build():
    S.use()
    X, Y = _data()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    cols = plt.get_cmap(S.CAT)(np.linspace(0.06, 0.76, len(ETAS)))

    for ax, norm, title in ((axes[0], None, "(a) no normalization"),
                            (axes[1], batch_norm,
                             "(b) with batch normalization")):
        for eta, col in zip(ETAS, cols):
            Ws = make_net(DIMS, lambda a, b: he_variance(a), seed=4)
            L, _ = train(Ws, X, Y, eta, 600, norm=norm, batch=64,
                         seed=4, norm_backward=batch_norm_backward)
            L = _smooth(np.nan_to_num(L, nan=1e9, posinf=1e9))
            L = np.clip(L, 1e-6, 5e3)
            ax.semilogy(L, color=col, lw=2.2, label=r"$\eta = %.2f$" % eta)
            if L[-1] >= 5e3 - 1:
                j = int(np.argmax(L >= 5e3 - 1))
                ax.annotate("diverges", xy=(j + 12, 1.6e3), fontsize=12.5,
                            color=col)
        ax.set_xlabel("iteration")
        ax.set_ylabel("training error")
        ax.set_ylim(1e-3, 8e3)
        ax.legend(loc="upper right", fontsize=13)
        ax.set_title(title)
        S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
