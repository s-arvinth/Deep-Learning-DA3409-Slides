#!/usr/bin/env python3
"""Universal approximation in D dimensions, by ridge functions.

Any continuous target on a compact set is a uniform limit of finite sums
of cos(w'x + b) -- each a RIDGE function, constant along the directions
orthogonal to w.  Each ridge term is a one-dimensional function of w'x,
so the D = 1 result then turns the whole sum into a one-hidden-layer
network.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import target_2d

NAME = "ua_ridge"


def ridge_design(P, W, Bv):
    """cos(w_k' x + b_k) evaluated on the grid points P (n, 2)."""
    return np.cos(P @ W.T + Bv)


def build():
    style.use()
    g = np.linspace(-2, 2, 220)
    X1, X2 = np.meshgrid(g, g)
    P = np.stack([X1.ravel(), X2.ravel()], 1)
    target = target_2d(X1, X2)
    t = target.ravel()

    rng = np.random.default_rng(3)
    K = 220
    W = rng.normal(0, 1.15, (K, 2))
    Bv = rng.uniform(0, 2 * np.pi, K)
    A = ridge_design(P, W, Bv)

    def fit(k):
        c, *_ = np.linalg.lstsq(A[:, :k], t, rcond=None)
        return (A[:, :k] @ c).reshape(X1.shape)

    vmin, vmax = target.min(), target.max()
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 8.8))

    m = ax[0, 0].pcolormesh(X1, X2, target, cmap=style.SEQ, shading="auto",
                            vmin=vmin, vmax=vmax)
    m.set_rasterized(True)
    ax[0, 0].set_title(r"(a)  target $f(\mathbf{x})$", fontsize=13)

    one = np.cos(P @ W[0] + Bv[0]).reshape(X1.shape)
    m1 = ax[0, 1].pcolormesh(X1, X2, one, cmap=style.SEQ, shading="auto")
    m1.set_rasterized(True)
    d = W[0] / np.linalg.norm(W[0])
    ax[0, 1].annotate("", xy=(1.5 * d[0], 1.5 * d[1]), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color="white",
                                      lw=2.6, mutation_scale=20))
    ax[0, 1].annotate(r"$\mathbf{w}_k$", xy=(1.6 * d[0], 1.6 * d[1]),
                      color="white", fontsize=15)
    ax[0, 1].set_title(r"(b)  one ridge term $\cos(\mathbf{w}_k^{T}"
                       r"\mathbf{x}+b_k)$", fontsize=11.5)

    m2 = ax[1, 0].pcolormesh(X1, X2, fit(K), cmap=style.SEQ, shading="auto",
                             vmin=vmin, vmax=vmax)
    m2.set_rasterized(True)
    err = np.abs(target - fit(K)).max()
    ax[1, 0].set_title(rf"(c)  $K={K}$ ridge terms,  "
                       rf"$\sup|f-\hat f|={err:.3f}$", fontsize=11.5)

    ks = np.unique(np.round(np.logspace(0.3, np.log10(K), 16)).astype(int))
    errs = [np.abs(target - fit(int(k))).max() for k in ks]
    a_ = ax[1, 1]
    a_.loglog(ks, errs, color=OUTC, lw=2.4, marker="o", ms=6, mec="white",
              mew=0.8, label=r"$\sup_{\mathbf{x}}|f-\hat f|$")
    a_.legend(loc="lower left", fontsize=12)
    a_.set(xlabel=r"number of ridge terms $K$", ylabel="maximum error")
    a_.set_title(r"(d)  error $\to 0$ as $K\to\infty$", fontsize=12.5)
    style.square(a_)

    for a2 in [ax[0, 0], ax[0, 1], ax[1, 0]]:
        a2.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
        a2.set_aspect("equal", "box")
    fig.colorbar(m, ax=ax[0, 0], fraction=0.046, pad=0.03)
    fig.colorbar(m1, ax=ax[0, 1], fraction=0.046, pad=0.03)
    fig.colorbar(m2, ax=ax[1, 0], fraction=0.046, pad=0.03)
    fig.tight_layout(w_pad=2.2, h_pad=2.6)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
