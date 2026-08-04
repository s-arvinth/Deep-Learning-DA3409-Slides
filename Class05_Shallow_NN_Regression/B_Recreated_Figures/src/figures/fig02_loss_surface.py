#!/usr/bin/env python3
"""The sum-of-squares error surface over (w0, w1), and gradient descent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import linear_data

NAME = "loss_surface"


def build():
    style.use()
    x, y = linear_data()
    g0 = np.linspace(-0.6, 1.4, 240)
    g1 = np.linspace(0.0, 2.4, 240)
    W0, W1 = np.meshgrid(g0, g1)
    E = np.zeros_like(W0)
    for xi, yi in zip(x, y):
        E += (W0 + W1 * xi - yi) ** 2
    E *= 0.5

    w = np.array([-0.4, 0.2]); path = [w.copy()]
    for _ in range(60):
        r = (w[0] + w[1] * x) - y
        w = w - 0.35 * np.array([2 * r.mean(), 2 * (r * x).mean()])
        path.append(w.copy())
    H = np.array(path)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.4))

    m = ax[0].pcolormesh(W0, W1, E, cmap=style.SEQ, shading="auto")
    m.set_rasterized(True)
    ax[0].contour(W0, W1, E, levels=12, colors="white", linewidths=0.6,
                  alpha=.6)
    cb = fig.colorbar(m, ax=ax[0], fraction=0.046, pad=0.03)
    cb.set_label(r"$E(\mathbf{w})$", fontsize=15.6)
    cb.ax.tick_params(labelsize=13.0)
    ax[0].set(xlabel=r"intercept  $w_0$", ylabel=r"slope  $w_1$",
              title=r"(a)  the error surface $E(\mathbf{w})$")

    ax[1].contour(W0, W1, E, levels=16, cmap=style.SEQ, linewidths=1.1)
    ax[1].plot(H[:, 0], H[:, 1], color=OUTC, lw=1.8, marker="o", ms=4.0,
               mec="white", mew=0.6, label="gradient descent")
    ax[1].scatter([H[0, 0]], [H[0, 1]], s=95, marker="s", color=L1,
                  edgecolor="white", zorder=5, label=r"start $\mathbf{w}^{(0)}$")
    ax[1].scatter([H[-1, 0]], [H[-1, 1]], s=150, marker="*", color=OUTC,
                  edgecolor="white", zorder=5,
                  label=r"minimum $\mathbf{w}_{\mathrm{ML}}$")
    ax[1].legend(loc="lower right", fontsize=13.0, framealpha=.93,
                 facecolor="white", edgecolor="none")
    ax[1].set(xlabel=r"intercept  $w_0$", ylabel=r"slope  $w_1$",
              title=r"(b)  one convex bowl, one minimum")
    for a in ax:
        style.square(a)
    fig.tight_layout(w_pad=2.4)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
