"""Unpooling, drawn.

The layout of Bishop & Bishop (2024), Figs. 10.28 and 10.29, in the
book's colours, every number produced by the pooling functions.

(a) The analogue of average pooling: each value is copied into its
    2 x 2 block.  Average-pooling the result gives the input back.
(b) The analogue of max pooling: each value goes to the first cell of
    its block and the rest are zero.  Max-pooling gives the input back.
(c) Remembering where each maximum was: a 4 x 4 array is max-pooled,
    passes through intermediate layers, and is unpooled with each value
    placed at the position its block's maximum came from.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import bishopdraw as D
from common.models import maxpool, avgpool, unpool_avg, unpool_max

NAME = "unpooling"
COL = {1: D.RED, 2: D.GREEN, 3: D.BLUE, 4: D.YELLOW}


def build():
    S.use()
    Z = np.array([[1, 2], [3, 4]])
    Za = unpool_avg(Z).astype(int); Zm = unpool_max(Z).astype(int)
    assert np.array_equal(avgpool(Za).astype(int), Z)
    assert np.array_equal(maxpool(Zm)[0].astype(int), Z)
    X = np.array([[5, 2, 4, 2], [7, 1, 0, 3], [3, 8, 9, 6], [4, 7, 8, 1]])
    P, where = maxpool(X); P = P.astype(int)
    Q = np.array([[3, 7], [9, 4]])                  # after intermediate layers
    U = unpool_max(Q, where=where).astype(int)
    print("(a)", Za.tolist(), "(b)", Zm.tolist(), "(c) pooled", P.tolist(),
          "unpooled", U.tolist())

    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.25, wspace=0.12)
    zc = [[COL[v] for v in row] for row in Z]
    # (a)
    ax = fig.add_subplot(gs[0, 0])
    D.flat_grid(ax, 0, 1, Z, zc); D.arrow(ax, 2.6, 4.2, 0)
    D.flat_grid(ax, 4.8, 2, Za, [[COL[v] for v in row] for row in Za])
    ax.set_xlim(-0.3, 9.1); ax.set_ylim(-2.3, 2.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) the analogue of average pooling", fontsize=12, loc="left")
    # (b)
    ax = fig.add_subplot(gs[0, 1])
    D.flat_grid(ax, 0, 1, Z, zc); D.arrow(ax, 2.6, 4.2, 0)
    D.flat_grid(ax, 4.8, 2, Zm, [[COL.get(v, D.WHITE) for v in row] for row in Zm])
    ax.set_xlim(-0.3, 9.1); ax.set_ylim(-2.3, 2.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) the analogue of max pooling", fontsize=12, loc="left")
    # (c)
    ax = fig.add_subplot(gs[1, :])
    cx = [[D.WHITE] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if where[i, j]:
                cx[i][j] = COL[2 * (i // 2) + j // 2 + 1]
    D.flat_grid(ax, 0, 2, X, cx); D.arrow(ax, 4.6, 6.4, 0)
    D.flat_grid(ax, 7.0, 1, P, zc)
    ax.text(10.9, 0.5, "intermediate layers", ha="center", va="bottom", fontsize=11)
    ax.plot(np.linspace(9.5, 12.3, 12), np.full(12, -0.1), ".", color="black", ms=3)
    D.flat_grid(ax, 12.8, 1, Q, zc); D.arrow(ax, 15.4, 17.2, 0)
    cu = [[D.WHITE] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if where[i, j]:
                cu[i][j] = COL[2 * (i // 2) + j // 2 + 1]
    D.flat_grid(ax, 17.8, 2, U, cu)
    ax.set_xlim(-0.3, 22.1); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(c) max-unpooling to the remembered positions", fontsize=12, loc="left")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
