"""Gradient descent on a convex loss, in parameter space and data space.

Recreates the structure of Prince (2023), Fig. 6.1 for the univariate
linear model, in the frozen course notation: the parameters are w_0 and
w_1 rather than Prince's phi.

Panel (a): the training set, with the fitted line at successive
iterations shading from light to dark.  Panel (b): the same iterations
on the loss surface, which for least squares is an elliptical bowl with
exactly one minimum.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "linreg_descent"


def build():
    S.use()
    rng = np.random.default_rng(3)
    n = 12
    x = np.linspace(0.05, 0.95, n)
    y = 0.28 + 0.86 * x + 0.055 * rng.standard_normal(n)

    def loss(w0, w1):
        return ((w0[..., None] + w1[..., None] * x - y) ** 2).sum(axis=-1)

    def grad(w):
        r = w[0] + w[1] * x - y
        return np.array([2.0 * r.sum(), 2.0 * (r * x).sum()])

    w = np.array([1.30, -0.55])
    path = [w.copy()]
    for _ in range(14):                     # line search, as Prince does
        g = grad(w)
        etas = np.linspace(0.0, 0.30, 400)[1:]
        cand = np.array([loss(np.array(w[0] - e * g[0]),
                              np.array(w[1] - e * g[1])) for e in etas])
        w = w - etas[int(cand.argmin())] * g
        path.append(w.copy())
    path = np.array(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    cols = plt.get_cmap(S.CAT)(np.linspace(0.82, 0.06, len(path)))

    ax = axes[0]
    xs = np.linspace(0.0, 1.0, 50)
    for k, (p, c) in enumerate(zip(path, cols)):
        if k % 2 and k != len(path) - 1:
            continue
        ax.plot(xs, p[0] + p[1] * xs, color=c, lw=2.0,
                zorder=2 + k,
                label=("start" if k == 0 else
                       ("after %d steps" % (len(path) - 1)
                        if k == len(path) - 1 else None)))
    ax.plot(x, y, "o", ms=8, mfc="white", mec=S.L1, mew=1.6, zorder=20)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_xlim(0, 1); ax.set_ylim(-0.30, 1.60)
    ax.legend(loc="lower right", fontsize=13)
    ax.set_title("(a) what the model does")
    S.square(ax)

    ax = axes[1]
    g0 = np.linspace(-0.35, 1.45, 260)
    g1 = np.linspace(-0.85, 1.85, 260)
    G0, G1 = np.meshgrid(g0, g1)
    Z = loss(G0, G1)
    ax.contourf(G0, G1, Z, levels=24, cmap=S.SEQ, alpha=0.92)
    ax.contour(G0, G1, Z, levels=24, colors="white", linewidths=0.4,
               alpha=0.55)
    ax.plot(path[:, 0], path[:, 1], "-", color="white", lw=1.8, zorder=4)
    for k, (p, c) in enumerate(zip(path, cols)):
        ax.plot(p[0], p[1], "o", ms=7.5, mfc=c, mec="white", mew=1.3,
                zorder=5)
        if k % 2 == 0 or k == len(path) - 1:
            ax.annotate(str(k), xy=p, xytext=(p[0] + 0.05, p[1] + 0.07),
                        color="white", fontsize=12)
    ax.set_xlabel("$w_0$"); ax.set_ylabel("$w_1$")
    ax.set_title("(b) where the parameters went")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
