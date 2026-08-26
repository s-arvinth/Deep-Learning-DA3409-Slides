"""Initialization is a statement about the first step only.

The weights are set so that every layer starts with unit variance.
Training then moves them, and after a few hundred updates the deeper
layers are no longer anywhere near unit variance. Whatever the
initialization achieved, it does not stay achieved.

(a) the standard deviation of the pre-activations, by layer, at four
points during training. (b) the same quantity for one layer, against
the iteration, with and without a normalization layer.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (make_net, forward, he_variance, batch_norm,
                           batch_norm_backward)

NAME = "drift"

DIMS = [40] + [40] * 8 + [1]
SNAP = [0, 60, 240, 900]


def _data(n=512, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, DIMS[0]))
    Y = np.sin(X[:, :1] * 1.3) + 0.4 * X[:, 1:2]
    return X, Y


def _train_track(norm, steps=901, eta=0.02, seed=1):
    X, Y = _data()
    Ws = make_net(DIMS, lambda a, b: he_variance(a), seed=seed)
    rng = np.random.default_rng(seed)
    snaps, series = {}, []
    for t in range(steps):
        _, acts = forward(Ws, X, norm=norm)
        sds = np.array([a.std() for a in acts[:-1]])
        series.append(sds)
        if t in SNAP:
            snaps[t] = sds.copy()
        idx = rng.choice(len(X), 64, replace=False)
        xb, yb = X[idx], Y[idx]
        A, cache = xb, []
        for k, W in enumerate(Ws):
            pre = A @ W.T
            post = norm(pre) if (norm is not None and k < len(Ws) - 1) else pre
            cache.append((A, pre, post))
            A = np.maximum(post, 0) if k < len(Ws) - 1 else post
        G = 2.0 * (A - yb) / len(xb)
        for k in range(len(Ws) - 1, -1, -1):
            Ain, pre, post = cache[k]
            if k < len(Ws) - 1:
                G = G * (post > 0)
                if norm is not None:
                    G = batch_norm_backward(G, pre)
            dW = G.T @ Ain
            G = G @ Ws[k]
            Ws[k] = Ws[k] - eta * dW
    return snaps, np.array(series)


def build():
    S.use()
    snaps, series = _train_track(None)
    _, series_bn = _train_track(batch_norm)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    cols = plt.get_cmap(S.CAT)(np.linspace(0.06, 0.78, len(SNAP)))
    for t, col in zip(SNAP, cols):
        ax.semilogy(np.arange(1, len(snaps[t]) + 1), snaps[t], "o-",
                    color=col, lw=2.1, ms=6, mfc="white", mew=1.5,
                    label="iteration %d" % t)
    ax.axhline(1.0, color=S.GREY, lw=1.3, ls=(0, (4, 3)))
    ax.annotate("where initialization put it", xy=(1.2, 1.25),
                fontsize=13, color=S.GREY)
    ax.set_xlabel("layer")
    ax.set_ylabel("s.d. of the pre-activations")
    ax.legend(loc="lower left", fontsize=12.5)
    ax.set_title("(a) the statistics drift as training proceeds")
    S.square(ax)

    ax = axes[1]
    L = series.shape[1] - 1
    ax.semilogy(series[:, L], color=S.OUTC, lw=2.2,
                label="no normalization")
    ax.semilogy(series_bn[:, L], color=S.L2, lw=2.2,
                label="with batch normalization")
    ax.axhline(1.0, color=S.GREY, lw=1.3, ls=(0, (4, 3)))
    ax.set_xlabel("iteration")
    ax.set_ylabel("s.d. at the last hidden layer")
    ax.legend(loc="lower left", fontsize=12.5)
    ax.set_title("(b) a normalizer holds it in place")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
