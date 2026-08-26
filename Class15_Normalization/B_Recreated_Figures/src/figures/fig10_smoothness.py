"""Why a normalization layer makes the surface easier to walk on.

Take the current parameters, choose the descent direction, and look at
the error along that direction over a range of step sizes. Without
normalization the error along the line varies wildly from one iteration
to the next, so the gradient measured here predicts very little about
what happens a step away. With normalization the same slice is far
better behaved.

(a) the spread of the error along the step direction, at each
iteration. (b) how well the gradient at the current point predicts the
gradient one step later.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (make_net, he_variance, batch_norm,
                           batch_norm_backward)

NAME = "smoothness"

DIMS = [30] + [30] * 5 + [1]
STEPS = 240


def _loss(Ws, X, Y, norm):
    A = X
    for k, W in enumerate(Ws):
        A = A @ W.T
        if norm is not None and k < len(Ws) - 1:
            A = norm(A)
        if k < len(Ws) - 1:
            A = np.maximum(A, 0)
    return float(((A - Y) ** 2).mean())


def _grad(Ws, X, Y, norm):
    A, cache = X, []
    for k, W in enumerate(Ws):
        pre = A @ W.T
        post = norm(pre) if (norm is not None and k < len(Ws) - 1) else pre
        cache.append((A, pre, post))
        A = np.maximum(post, 0) if k < len(Ws) - 1 else post
    G = 2.0 * (A - Y) / len(X)
    gs = [None] * len(Ws)
    for k in range(len(Ws) - 1, -1, -1):
        Ain, pre, post = cache[k]
        if k < len(Ws) - 1:
            G = G * (post > 0)
            if norm is not None:
                G = batch_norm_backward(G, pre)
        gs[k] = G.T @ Ain
        G = G @ Ws[k]
    return gs


def _run(norm, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((256, DIMS[0]))
    Y = np.sin(1.3 * X[:, :1])
    Ws = make_net(DIMS, lambda a, b: he_variance(a), seed=seed)
    etas = np.linspace(0.0, 0.06, 11)
    spread, align = [], []
    prev = None
    for t in range(STEPS):
        gs = _grad(Ws, X, Y, norm)
        vals = []
        for e in etas:
            Wt = [W - e * g for W, g in zip(Ws, gs)]
            vals.append(_loss(Wt, X, Y, norm))
        vals = np.array(vals)
        base = max(vals[0], 1e-12)
        spread.append((vals.max() - vals.min()) / base)
        flat = np.concatenate([g.ravel() for g in gs])
        if prev is not None:
            c = (prev @ flat) / (np.linalg.norm(prev)
                                 * np.linalg.norm(flat) + 1e-12)
            align.append(c)
        prev = flat
        Ws = [W - 0.02 * g for W, g in zip(Ws, gs)]
    return np.array(spread), np.array(align)


def _smooth(x, k=11):
    return np.array([np.median(x[max(0, i - k):i + 1])
                     for i in range(len(x))])


def build():
    S.use()
    sp0, al0 = _run(None)
    sp1, al1 = _run(batch_norm)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    ax.semilogy(np.clip(_smooth(sp0), 1e-4, 30), color=S.OUTC, lw=2.2,
                label="no normalization")
    ax.semilogy(np.clip(_smooth(sp1), 1e-4, 30), color=S.L2, lw=2.2,
                label="with batch normalization")
    ax.set_ylim(1e-3, 60)
    ax.set_xlabel("iteration")
    ax.set_ylabel("spread of $E$ along the step direction")
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(a) how much $E$ varies over one step")
    S.square(ax)

    ax = axes[1]
    ax.plot(_smooth(al0), color=S.OUTC, lw=2.2, label="no normalization")
    ax.plot(_smooth(al1), color=S.L2, lw=2.2,
            label="with batch normalization")
    ax.axhline(1.0, color=S.GREY, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("iteration")
    ax.set_ylabel("cosine between successive gradients")
    ax.set_ylim(-0.35, 1.15)
    ax.legend(loc="lower right", fontsize=13)
    ax.set_title("(b) how far the gradient can be trusted")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
