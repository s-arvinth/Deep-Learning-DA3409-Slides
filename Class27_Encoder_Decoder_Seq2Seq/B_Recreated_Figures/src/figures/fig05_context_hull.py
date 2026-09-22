"""The context is a point in the convex hull of the encoder states.

The encoder states h^e_1..h^e_n of the attention model on one source,
projected to two dimensions by PCA (the projection is linear, so
convex combinations project to convex combinations).  Each attention
context c_i = sum_j alpha_ij h^e_j is a convex combination and lies
inside the hull; with alpha concentrated on one position, c_i sits
near that vertex.  The fixed context of the plain model is the single
vertex h^e_n, whatever the decoder needs.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.experiment import run

NAME = "context_hull"
SRC = np.array([2, 7, 1, 9, 3, 8, 5, 4, 6, 1])


def hull(P):
    """Andrew's monotone chain on 2-D points."""
    P = sorted(map(tuple, P))
    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1]) - (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo = half(P); hi = half(P[::-1])
    return np.array(lo[:-1] + hi[:-1])


def build():
    S.use()
    m = run()["attention"]["model"]
    H = np.array([Hj.v[0] for Hj in m.encode(SRC[None, :])])          # (n, d)
    _, alphas = m.greedy_decode(SRC[None, :]); A = alphas[0]
    C = A @ H                                                          # contexts
    mu = H.mean(0); U, s, Vt = np.linalg.svd(H - mu, full_matrices=False)
    P = (H - mu) @ Vt[:2].T; Pc = (C - mu) @ Vt[:2].T
    hp = hull(P)
    print("variance explained by two components: %.0f%%" % (100 * (s[:2] ** 2).sum() / (s ** 2).sum()))

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.fill(hp[:, 0], hp[:, 1], color=S.L1, alpha=0.08, lw=0)
    ax.plot(np.r_[hp[:, 0], hp[0, 0]], np.r_[hp[:, 1], hp[0, 1]], color=S.L1, lw=1.2)
    ax.plot(P[:, 0], P[:, 1], "o", ms=8, mfc="white", mec=S.L1, mew=1.8, zorder=5, label="encoder states $\\mathbf{h}^e_j$")
    for j, (x, y) in enumerate(P):
        ax.annotate("%d" % (j + 1), xy=(x, y), xytext=(6, 4), textcoords="offset points", fontsize=10, color=S.L1)
    ax.plot(Pc[:-1, 0], Pc[:-1, 1], "s", ms=6, color=S.L2, zorder=6, label="attention contexts $\\mathbf{c}_i$")
    ax.plot(P[-1, 0], P[-1, 1], "*", ms=16, mfc=S.OUTC, mec="white", zorder=7, label="fixed context $\\mathbf{c} = \\mathbf{h}^e_n$")
    ax.set_xlabel("first principal direction"); ax.set_ylabel("second")
    ax.legend(fontsize=10.5, loc="best")
    ax.set_title("contexts live inside the hull of the encoder states")
    S.square(ax)
    fig.tight_layout()
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
