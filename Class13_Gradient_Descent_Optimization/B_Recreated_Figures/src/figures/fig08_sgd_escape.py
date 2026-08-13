"""Noise in the estimate is what lets the iterate change valleys.

Recreates the comparison of Prince (2023), Fig. 6.5: the same start on
the same non-convex loss, run with the full-batch gradient and with a
mini-batch gradient.  Full batch descends the valley it was dropped in.
Mini-batch occasionally steps uphill on the true loss and can leave.

The third panel makes the mechanism explicit in Prince's own reading of
SGD (his Fig. 6.6): each batch defines a *different* loss surface, and
the algorithm does exact descent on a surface that keeps changing.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import gabor_data, gabor_loss, gabor_grad

NAME = "sgd_escape"

W0 = np.linspace(-9.0, 9.0, 300)
W1 = np.linspace(2.0, 30.0, 300)
START = np.array([2.4, 6.2])
STEP = np.array([0.030, 0.075])


def _surface(x, y):
    G0, G1 = np.meshgrid(W0, W1)
    Z = np.empty_like(G0)
    for j in range(G0.shape[0]):
        Z[j] = gabor_loss(G0[j], G1[j], x, y)
    return G0, G1, Z


def _run(x, y, batch, steps, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    w = START.copy()
    path = [w.copy()]
    order = rng.permutation(n); k = 0
    for _ in range(steps):
        if batch >= n:
            g = gabor_grad(w, x, y)
        else:
            if k + batch > n:
                order = rng.permutation(n); k = 0
            idx = order[k:k + batch]; k += batch
            g = gabor_grad(w, x[idx], y[idx]) * (n / batch)
        g = g / max(1.0, np.linalg.norm(g) / 6.0)
        w = w - STEP * g
        path.append(w.copy())
    return np.array(path)


def build():
    S.use()
    x, y = gabor_data()
    G0, G1, Z = _surface(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))

    for ax, (batch, title) in zip(axes[:2],
                                  [(len(x), "(a) full batch"),
                                   (3, "(b) batch of 3")]):
        ax.contourf(G0, G1, Z, levels=24, cmap=S.SEQ, alpha=0.94)
        ax.contour(G0, G1, Z, levels=24, colors="white", linewidths=0.35,
                   alpha=0.5)
        seeds = [0] if batch >= len(x) else [1, 5, 13]
        for sd, col in zip(seeds, [S.OUTC, S.ACC, "#F2F2F2"]):
            p = _run(x, y, batch, 700, sd)
            ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2.0, zorder=4)
            ax.plot(p[-1, 0], p[-1, 1], "s", ms=8, mfc="white", mec=col,
                    mew=2.0, zorder=6)
        ax.plot(*START, "o", ms=10, mfc="white", mec=S.OUTC, mew=2.2,
                zorder=7)
        ax.set_xlabel("$w_0$"); ax.set_ylabel("$w_1$")
        ax.set_title(title)
        S.square(ax)

    # ---- (c) a batch defines its own surface --------------------------
    # Prince's reading of SGD (his Fig. 6.6): the update is exact descent
    # on a surface that is redrawn at every iteration.  Along one slice
    # of parameter space, three batches disagree about where downhill is.
    ax = axes[2]
    rng = np.random.default_rng(4)
    w1_slice = 16.6
    ax.plot(W0, gabor_loss(W0, np.full_like(W0, w1_slice), x, y),
            color=S.GREY, lw=2.6, zorder=5, label="full training set")
    cols = plt.get_cmap(S.CAT)(np.linspace(0.08, 0.72, 3))
    for k, c in enumerate(cols):
        idx = rng.choice(len(x), size=3, replace=False)
        Lb = (gabor_loss(W0, np.full_like(W0, w1_slice), x[idx], y[idx])
              * len(x) / 3.0)
        ax.plot(W0, Lb, color=c, lw=1.7, alpha=0.95,
                label="a batch of 3" if k == 0 else None)
        j = int(np.argmin(Lb))
        ax.plot(W0[j], Lb[j], "v", ms=9, mfc=c, mec="white", mew=1.2,
                zorder=6)
    j = int(np.argmin(gabor_loss(W0, np.full_like(W0, w1_slice), x, y)))
    ax.axvline(W0[j], color=S.GREY, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("$w_0$   (slice at $w_1 = 16.6$)")
    ax.set_ylabel("$E(w)$")
    ax.set_ylim(0, None)
    ax.legend(loc="upper center", fontsize=12.5)
    ax.set_title("(c) each batch draws its own surface")
    S.square(ax)

    fig.tight_layout(w_pad=1.4)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
