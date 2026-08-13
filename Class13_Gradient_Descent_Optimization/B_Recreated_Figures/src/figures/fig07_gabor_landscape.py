"""A non-convex loss, and where gradient descent ends up on it.

Recreates the argument of Prince (2023), Figs. 6.4 and 6.5a using the
Gabor model of Prince Eq. 6.8.  Two parameters is the largest number we
can draw, and it is already enough to show that the destination of
gradient descent is decided by the starting point.

Panel (a) is the loss surface with its local minima, its global minimum
and a saddle marked.  Panel (b) runs full-batch gradient descent from
three different starts.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import gabor_data, gabor_loss, gabor_grad

NAME = "gabor_landscape"

W0 = np.linspace(-9.0, 9.0, 340)
W1 = np.linspace(2.0, 30.0, 340)
STARTS = [(-6.0, 8.5), (2.4, 6.2), (5.6, 24.0)]


def _local_minima(G0, G1, Z, half=13):
    """Grid points that are the smallest in a (2*half+1)-square window.

    A plain sliding-window minimum, written out rather than imported, so
    the figure has no dependency beyond numpy and matplotlib.
    """
    out = []
    ny, nx = Z.shape
    for j in range(half, ny - half):
        for i in range(half, nx - half):
            win = Z[j - half:j + half + 1, i - half:i + half + 1]
            if Z[j, i] <= win.min():
                out.append((G0[j, i], G1[j, i], Z[j, i]))
    # thin out near-duplicates from flat regions
    keep = []
    for p in sorted(out, key=lambda t: t[2]):
        if all((p[0] - q[0]) ** 2 + ((p[1] - q[1]) / 4.5) ** 2 > 0.9
               for q in keep):
            keep.append(p)
    return keep


def _grid(x, y):
    G0, G1 = np.meshgrid(W0, W1)
    Z = np.empty_like(G0)
    for j in range(G0.shape[0]):
        Z[j] = gabor_loss(G0[j], G1[j], x, y)
    return G0, G1, Z


def build():
    S.use()
    x, y = gabor_data()
    G0, G1, Z = _grid(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))

    # ---- (a) the landscape -------------------------------------------
    ax = axes[0]
    ax.contourf(G0, G1, Z, levels=26, cmap=S.SEQ, alpha=0.94)
    ax.contour(G0, G1, Z, levels=26, colors="white", linewidths=0.35,
               alpha=0.5)

    # locate the minima on the grid, then mark the best one separately
    pts = _local_minima(G0, G1, Z, half=9)
    pts.sort(key=lambda t: t[2])
    best = pts[0]
    for p in pts[1:11]:
        ax.plot(p[0], p[1], "o", ms=10, mfc="none", mec="#7FE3F0", mew=2.4,
                zorder=5)
    ax.plot(best[0], best[1], "*", ms=18, mfc="white", mec=S.OUTC, mew=1.8,
            zorder=6)
    ax.plot([], [], "o", ms=9, mfc="none", mec="#7FE3F0", mew=2.2,
            label="local minima")
    ax.plot([], [], "*", ms=14, mfc="white", mec=S.OUTC, mew=1.6,
            label="global minimum")
    ax.legend(loc="upper right", fontsize=13)
    ax.set_xlabel("$w_0$"); ax.set_ylabel("$w_1$")
    ax.set_title("(a) the loss is not convex")
    S.square(ax)

    # ---- (b) three starts, three destinations ------------------------
    ax = axes[1]
    ax.contourf(G0, G1, Z, levels=26, cmap=S.SEQ, alpha=0.94)
    ax.contour(G0, G1, Z, levels=26, colors="white", linewidths=0.35,
               alpha=0.5)
    cols = [S.OUTC, S.ACC, "#F2F2F2"]
    for (s0, s1), col in zip(STARTS, cols):
        w = np.array([s0, s1], float)
        path = [w.copy()]
        for _ in range(700):
            g = gabor_grad(w, x, y)
            g = g / max(1.0, np.linalg.norm(g) / 6.0)
            w = w - np.array([0.030, 0.075]) * g
            path.append(w.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-", color=col, lw=2.2, zorder=4)
        ax.plot(path[0, 0], path[0, 1], "o", ms=9, mfc=col, mec="white",
                mew=1.5, zorder=6)
        ax.plot(path[-1, 0], path[-1, 1], "s", ms=8, mfc="white", mec=col,
                mew=2.0, zorder=6)
    ax.set_xlabel("$w_0$"); ax.set_ylabel("$w_1$")
    ax.set_title("(b) the start decides the finish")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
