"""Why the weights cannot all start at the same value.

Hidden units that receive the same inputs and start with the same
weights compute the same function, receive the same gradient, and are
therefore updated identically forever. The layer has the capacity of a
single unit no matter how many units it contains.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "symmetry"

M, T = 6, 60


def _run(identical, seed=0):
    """Fit y = sum_j w_j tanh(v_j x) with M hidden units."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.5, 2.5, 120)
    y = np.sin(1.7 * x)
    if identical:
        v = np.full(M, 0.5); w = np.full(M, 0.45)
    else:
        v = rng.normal(0, 0.9, M); w = rng.normal(0, 0.5, M)
    hist = [v.copy()]
    for _ in range(T):
        h = np.tanh(np.outer(x, v))
        pred = h @ w
        r = 2.0 * (pred - y) / len(x)
        gw = h.T @ r
        gv = ((r[:, None] * w) * (1 - h ** 2) * x[:, None]).sum(axis=0)
        w = w - 0.9 * gw
        v = v - 0.9 * gv
        hist.append(v.copy())
    return np.array(hist)


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    cols = plt.get_cmap(S.CAT)(np.linspace(0.05, 0.80, M))

    for ax, ident, title in ((axes[0], True, "(a) all units start equal"),
                             (axes[1], False, "(b) random initialization")):
        H = _run(ident)
        for j in range(M):
            ax.plot(H[:, j], color=cols[j], lw=2.0,
                    alpha=0.95 if not ident else 0.85)
        ax.set_xlabel("iteration")
        ax.set_ylabel("input weight of each hidden unit")
        ax.set_ylim(-2.4, 2.4)
        ax.set_title(title)
        S.square(ax)
    H0 = _run(True)
    axes[0].annotate("all six curves coincide:\nsix units, one function",
                     xy=(34, H0[34, 0]), xytext=(30, 1.70), fontsize=13.5,
                     ha="center", color=S.OUTC,
                     arrowprops=dict(arrowstyle="-|>", color=S.OUTC,
                                     lw=1.3))

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
