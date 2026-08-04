#!/usr/bin/env python3
"""Fixed basis functions vs learned hidden units: both linearise the problem."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import noisy_xor
from common.models import (gaussian_basis, fit_logistic, sigmoid,
                           fit_shallow_classifier)

NAME = "feature_space"


def build():
    style.use()
    X, y = noisy_xor(N=55, spread=0.30)
    centres = np.array([[-1.0, -1.0], [1.0, 1.0]])

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.3))
    cols = [L1, OUTC]; mk = ["o", "s"]

    # (a) input space
    for c in (0, 1):
        ax[0].scatter(X[y == c, 0], X[y == c, 1], s=26, marker=mk[c],
                      facecolor="white", edgecolor=cols[c], lw=1.2,
                      label=rf"$\mathcal{{C}}_{c+1}$")
    ax[0].scatter(centres[:, 0], centres[:, 1], s=150, marker="X",
                  color=L2, edgecolor="white", lw=1.2, zorder=6)
    for c in centres:
        ax[0].add_patch(plt.Circle(c, 1.15, fill=False, ec=L2, lw=1.2,
                                   ls="--", alpha=.8))
    ax[0].legend(loc="upper left", fontsize=13.0, framealpha=.92,
                 facecolor="white", edgecolor="none")
    ax[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$", xlim=(-2.4, 2.4),
              ylim=(-2.4, 2.4))
    ax[0].set_title(r"(a)  input space: no straight line works",
                    fontsize=15.0)

    # (b) fixed Gaussian basis functions
    Phi = gaussian_basis(X, centres, s=1.15)
    w = fit_logistic(Phi, y)
    g = np.linspace(-0.05, 1.15, 100)
    for c in (0, 1):
        ax[1].scatter(Phi[y == c, 0], Phi[y == c, 1], s=26, marker=mk[c],
                      facecolor="white", edgecolor=cols[c], lw=1.2)
    ax[1].plot(g, -(w[0] + w[1] * g) / w[2], color="black", lw=2.0)
    ax[1].set(xlabel=r"$\phi_1(\mathbf{x})$", ylabel=r"$\phi_2(\mathbf{x})$",
              xlim=(-0.05, 1.1), ylim=(-0.05, 1.1))
    ax[1].set_title(r"(b)  fixed features: linearly separable",
                    fontsize=15.0)

    # (c) learned hidden units
    W1, b1, W2, b2 = _train(X, y)
    Z = np.tanh(X @ W1.T + b1)
    for c in (0, 1):
        ax[2].scatter(Z[y == c, 0], Z[y == c, 1], s=26, marker=mk[c],
                      facecolor="white", edgecolor=cols[c], lw=1.2)
    zg = np.linspace(-1.1, 1.1, 100)
    if abs(W2[1]) > 1e-6:
        ax[2].plot(zg, -(b2 + W2[0] * zg) / W2[1], color="black", lw=2.0)
    ax[2].set(xlabel=r"$z_1$", ylabel=r"$z_2$",
              xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
    ax[2].set_title(r"(c)  learned features ($\tanh$): also linear",
                    fontsize=15.0)

    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


def _train(X, y, M=2, steps=9000, lr=0.9):
    """Two tanh hidden units + sigmoid output, cross-entropy, full batch.

    Retries seeds until both units are alive and the fit is essentially
    perfect, so the picture is never a degenerate collapse.
    """
    best = None
    N = len(X)
    for seed in range(60):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(0, 1.0, (M, X.shape[1])); b1 = np.zeros(M)
        W2 = rng.normal(0, 1.0, M); b2 = 0.0
        for _ in range(steps):
            A1 = X @ W1.T + b1
            Z = np.tanh(A1)
            yhat = sigmoid(Z @ W2 + b2)
            d = (yhat - y) / N
            gW2 = Z.T @ d; gb2 = d.sum()
            dZ = np.outer(d, W2) * (1 - Z ** 2)
            W1 -= lr * (dZ.T @ X); b1 -= lr * dZ.sum(0)
            W2 -= lr * gW2;        b2 -= lr * gb2
        Z = np.tanh(X @ W1.T + b1)
        acc = ((sigmoid(Z @ W2 + b2) > 0.5) == (y > 0.5)).mean()
        if (Z.std(0) > 0.05).all() and acc > 0.98:
            return W1, b1, W2, b2
        if best is None or acc > best[0]:
            best = (acc, (W1, b1, W2, b2))
    return best[1]


if __name__ == "__main__":
    build()
