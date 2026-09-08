"""Finite data does not determine a function.

Bishop & Bishop (2024), Section 9.1.1 calls this an inverse problem:
infinitely many mappings are consistent with the observations, so a
learning algorithm must prefer some of them, and that preference is what
the rest of this class is about.

(a) Eight observations and five functions, each passing through every
    one of them. Nothing in the data separates them.
(b) The same eight observations with a penalty on the weights. One
    function is now singled out, and which one depends on the penalty.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import truth, design

NAME = "underdetermined"

GRID = np.linspace(0.0, 1.0, 700)
XS = np.array([0.04, 0.15, 0.27, 0.39, 0.55, 0.68, 0.82, 0.96])
K = 60


def build():
    S.use()
    ys = truth(XS)
    Phi = design(XS, K)
    Phig = design(GRID, K)
    w0 = np.linalg.pinv(Phi) @ ys                 # minimum-norm interpolant
    U, s, Vt = np.linalg.svd(Phi)
    null = Vt[len(s):]                            # directions the data ignores

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))

    # ---- (a) many exact fits ----------------------------------------
    ax = axes[0]
    rng = np.random.default_rng(0)
    cols = plt.get_cmap(S.CAT)(np.linspace(0.08, 0.78, 5))
    for c, col in zip(range(5), cols):
        v = null[rng.integers(0, len(null), size=6)].sum(axis=0)
        df = Phig @ v
        v = v * (1.15 / np.max(np.abs(df)))      # scale in function space
        ax.plot(GRID, Phig @ (w0 + v), color=col, lw=1.9, zorder=4)
    ax.plot(XS, ys, "o", ms=8.5, mfc="white", mec=S.L1, mew=2.1, zorder=6)
    ax.annotate("all five fit every point exactly", xy=(0.5, 0.045),
                xycoords="axes fraction", ha="center", fontsize=13,
                color=S.GREY)
    ax.set_title("(a) the data does not choose")

    # ---- (b) the penalty chooses ------------------------------------
    ax = axes[1]
    for lam, col, lab in ((1e-8, S.GREY, r"$\lambda \to 0$"),
                          (3e-1, S.L2, r"$\lambda = 0.3$"),
                          (2e1, S.OUTC, r"$\lambda = 20$")):
        w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]),
                            Phi.T @ ys)
        ax.plot(GRID, Phig @ w, color=col, lw=2.4, zorder=4, label=lab)
    ax.plot(XS, ys, "o", ms=8.5, mfc="white", mec=S.L1, mew=2.1, zorder=6)
    ax.legend(loc="lower center", fontsize=12.5, ncol=3, handlelength=1.2,
              columnspacing=1.0)
    ax.set_title("(b) a penalty does")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_box_aspect(0.80)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
