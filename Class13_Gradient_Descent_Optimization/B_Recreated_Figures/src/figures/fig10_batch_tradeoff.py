"""What the batch size actually buys, measured two ways.

Panel (a): trajectories on a quadratic at three batch sizes.  Small
batches wander and settle into a noise ball whose radius grows as the
batch shrinks; the full batch walks a smooth curve to the minimum.

Panel (b): the same three runs plotted against *gradient evaluations*
rather than iterations, which is the axis that costs money.  Per unit of
computation the small batch is far ahead early and behind late -- which
is why practice uses a mini-batch rather than either extreme.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quadratic_E

NAME = "batch_tradeoff"

LAM = (1.0, 5.0)
N = 512
ETA = 0.055


def _run(batch, steps, seed):
    rng = np.random.default_rng(seed)
    w = np.array([-2.05, 1.75])
    H = np.diag(LAM)
    # per-example gradients = true gradient plus zero-mean example noise
    noise = rng.normal(scale=2.4, size=(N, 2))
    noise -= noise.mean(axis=0)
    path, cost = [w.copy()], [0]
    order = rng.permutation(N); k = 0
    for t in range(steps):
        if batch >= N:
            g = H @ w
        else:
            if k + batch > N:
                order = rng.permutation(N); k = 0
            idx = order[k:k + batch]; k += batch
            g = H @ w + noise[idx].mean(axis=0)
        w = w - ETA * g
        path.append(w.copy())
        cost.append(cost[-1] + min(batch, N))
    return np.array(path), np.array(cost)


def build():
    S.use()
    g = np.linspace(-2.6, 2.6, 260)
    G1, G2 = np.meshgrid(g, g)
    Z = quadratic_E(G1, G2, lam=LAM)

    settings = [(1, S.OUTC, "$B = 1$"),
                (16, S.ACC, "$B = 16$"),
                (N, S.L1, "full batch")]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    ax.contour(G1, G2, Z, levels=np.array([0.4, 1.2, 2.6, 4.8, 7.6]),
               colors=[S.GREY], linewidths=0.8, alpha=0.55)
    for (batch, col, lab), z in zip(settings, (3, 5, 7)):
        p, _ = _run(batch, 140, seed=2)
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.5,
                alpha=0.55 if batch == 1 else 1.0, label=lab, zorder=z)
    ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8, zorder=8)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(a) trajectories, per iteration")
    S.square(ax)

    ax = axes[1]
    for batch, col, lab in settings:
        p, c = _run(batch, 26000 if batch == 1 else (1700 if batch == 16 else 60), seed=2)
        E = 0.5 * (LAM[0] * p[:, 0] ** 2 + LAM[1] * p[:, 1] ** 2)
        # a running median, so the small-batch curve shows its level
        # rather than the spikes of individual draws
        k = 61 if batch == 1 else (9 if batch == 16 else 1)
        if k > 1:
            E = np.array([np.median(E[max(0, i - k):i + 1])
                          for i in range(len(E))])
        m = c <= 24000
        ax.semilogy(c[m], E[m], color=col, lw=2.0, label=lab)
    ax.set_xlabel("gradient evaluations")
    ax.set_ylabel(r"$E(\mathbf{w})$   (running median)")
    ax.set_xlim(0, 24000)
    ax.set_ylim(1e-4, 20)
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(b) the same runs, per unit of work")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
