#!/usr/bin/env python3
"""Representation learning: the network rearranges the data until a line works."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import two_moons
from common.models import relu

NAME = "representation"


def _train(seed, steps=6000, lr=0.5):
    rng = np.random.default_rng(seed)
    X, Y = two_moons(200, 0.10, seed)
    Xs = (X - X.mean(0)) / X.std(0)
    M1, M2 = 24, 2
    W1 = rng.normal(0, 1.0, (2, M1));               b1 = np.zeros(M1)
    W2 = rng.normal(0, np.sqrt(2 / M1), (M1, M2));  b2 = np.full(M2, 0.1)
    W3 = rng.normal(0, np.sqrt(2 / M2), (M2, 1));   b3 = np.zeros(1)
    hist = []
    for t in range(steps):
        Z1 = relu(Xs @ W1 + b1)
        Z2 = relu(Z1 @ W2 + b2)
        a = (Z2 @ W3 + b3).ravel()
        p = 1 / (1 + np.exp(-a))
        if t % 50 == 0:
            hist.append(-np.mean(Y * np.log(p + 1e-9) +
                                 (1 - Y) * np.log(1 - p + 1e-9)))
        d = (p - Y) / len(Y)
        gW3 = Z2.T @ d[:, None]; gb3 = d.sum(keepdims=True)
        dZ2 = d[:, None] @ W3.T * (Z2 > 0)
        gW2 = Z1.T @ dZ2;        gb2 = dZ2.sum(0)
        dZ1 = dZ2 @ W2.T * (Z1 > 0)
        gW1 = Xs.T @ dZ1;        gb1 = dZ1.sum(0)
        for arr, gr in ((W3, gW3), (b3, gb3), (W2, gW2),
                        (b2, gb2), (W1, gW1), (b1, gb1)):
            arr -= lr * gr
    Z1 = relu(Xs @ W1 + b1); Z2 = relu(Z1 @ W2 + b2)
    a = (Z2 @ W3 + b3).ravel()
    acc = ((a > 0).astype(float) == Y).mean()
    return X, Y, Z2, W3, b3, acc, hist, (Z2.std(0) > 1e-3).all()


def build():
    style.use()
    for seed in range(25):
        X, Y, Z2, W3, b3, acc, hist, ok = _train(seed)
        if ok and acc > 0.97:
            break

    fig, ax = plt.subplots(1, 3, figsize=(11.6, 4.3))
    ax[0].scatter(*X[Y == 0].T, s=34, marker="o", facecolor=L1,
                  edgecolor="w", lw=0.6, alpha=.9, label=r"class $0$")
    ax[0].scatter(*X[Y == 1].T, s=38, marker="^", facecolor=OUTC,
                  edgecolor="w", lw=0.6, alpha=.9, label=r"class $1$")
    ax[0].legend(loc="upper right", fontsize=11)
    ax[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax[0].set_title(r"(a)  input space: not linearly separable",
                    fontsize=11.5)
    ax[0].set_aspect("equal", "box")

    ax[1].scatter(*Z2[Y == 0].T, s=34, marker="o", facecolor=L1,
                  edgecolor="w", lw=0.6, alpha=.9)
    ax[1].scatter(*Z2[Y == 1].T, s=38, marker="^", facecolor=OUTC,
                  edgecolor="w", lw=0.6, alpha=.9)
    zx = np.linspace(Z2[:, 0].min(), Z2[:, 0].max(), 60)
    if abs(W3[1, 0]) > 1e-9:
        ax[1].plot(zx, -(W3[0, 0] * zx + b3[0]) / W3[1, 0], color="k",
                   lw=1.6, ls="--",
                   label=r"$\mathbf{w}^{T}\mathbf{z}+w_0=0$")
        ax[1].legend(loc="upper right", fontsize=10.5)
    ax[1].set(xlabel=r"$z^{(2)}_1$", ylabel=r"$z^{(2)}_2$")
    ax[1].set_title(rf"(b)  last hidden layer ({acc*100:.0f}\% correct)",
                    fontsize=11.5)
    style.square(ax[1])

    ax[2].plot(np.arange(len(hist)) * 50, hist, color=L2, lw=2.4)
    ax[2].set(xlabel="gradient-descent step",
              ylabel="cross-entropy error")
    ax[2].set_title(r"(c)  training error", fontsize=11.5)
    style.square(ax[2])
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
