"""Rescaling the inputs rescales the error surface.

(a) two features on very different scales, and the same data after each
has been given zero mean and unit variance. (b) the least-squares error
surface for a linear model on the raw features: a long narrow valley.
(c) the same surface after normalization: circular contours, and one
learning rate that suits both directions.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (two_feature_data, standardize, linear_hessian,
                           condition_number)

NAME = "input_scaling"


def build():
    S.use()
    X = two_feature_data(n=300, scale=(1.0, 9.0), corr=0.0, seed=3)
    Xn, _, _ = standardize(X)
    w_true = np.array([1.0, 0.35])
    rng = np.random.default_rng(1)
    y = X @ w_true + 0.4 * rng.standard_normal(len(X))
    yn = y.copy()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))

    # ---- (a) the data -------------------------------------------------
    ax = axes[0]
    ax.plot(X[:, 0], X[:, 1], "o", ms=5.5, mfc="none", mec=S.OUTC,
            mew=1.1, label="raw")
    ax.plot(Xn[:, 0], Xn[:, 1], "x", ms=5.5, color=S.L2, mew=1.3,
            label="normalized")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_xlim(-30, 30); ax.set_ylim(-30, 30)
    ax.legend(loc="upper left", fontsize=13)
    ax.set_title("(a) the same data, twice")
    S.square(ax)

    # ---- (b), (c) the error surfaces ---------------------------------
    for ax, D, t, yy in ((axes[1], X, "(b) raw features", y),
                         (axes[2], Xn, "(c) after normalization", yn)):
        wls = np.linalg.lstsq(D, yy, rcond=None)[0]
        H = linear_hessian(D)
        k = condition_number(H)
        span = 2.6 / np.sqrt(np.linalg.eigvalsh(H).min())
        g1 = np.linspace(wls[0] - span, wls[0] + span, 220)
        g2 = np.linspace(wls[1] - span, wls[1] + span, 220)
        G1, G2 = np.meshgrid(g1, g2)
        Z = np.empty_like(G1)
        for j in range(G1.shape[0]):
            W = np.stack([G1[j], G2[j]], axis=1)
            Z[j] = ((W @ D.T - yy) ** 2).mean(axis=1)
        ax.contour(G1, G2, Z, levels=np.geomspace(Z.min() + 1e-6,
                                                  Z.max(), 9)[1:7],
                   colors=[S.GREY], linewidths=0.9, alpha=0.65)
        # gradient descent from a common relative start
        eta = 1.0 / np.linalg.eigvalsh(H).max()
        w = wls + np.array([0.72 * span, 0.72 * span])
        p = [w.copy()]
        for _ in range(60):
            w = w - eta * (H @ (w - wls))
            p.append(w.copy())
        p = np.array(p)
        ax.plot(p[:, 0], p[:, 1], "-o", color=S.OUTC, lw=1.5, ms=3.8,
                mfc="white", mew=0.9, zorder=4)
        ax.plot(*wls, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8,
                zorder=6)
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
        ax.set_title("%s\n$\\kappa = %.0f$" % (t, k), fontsize=15.5)
        S.square(ax)

    fig.tight_layout(w_pad=1.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
