"""Why the data has to be shuffled.

Bishop & Bishop (2024), Section 7.2.4 warns that raw data sets often
carry an ordering -- collected by date, sorted alphabetically, grouped by
class -- and that drawing mini-batches as consecutive blocks then makes
every batch unrepresentative.

Panel (a) shows what a batch of eight contains under the two regimes
when the file happens to be sorted by class.  Panel (b) is the training
curve that results: the sorted run oscillates with the period of the
class blocks, because each batch pulls the parameters toward whichever
class it happens to contain.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "shuffling"

N, K, B = 480, 4, 8


def _curve(shuffled, seed=0):
    """Logistic-style toy: four class means, one linear parameter each."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(K), N // K)
    targets = np.array([-1.4, -0.4, 0.6, 1.6])
    y = targets[labels] + 0.25 * rng.standard_normal(N)
    w = np.zeros(K)
    order = rng.permutation(N) if shuffled else np.arange(N)
    losses, k = [], 0
    for _ in range(320):
        if k + B > N:
            order = rng.permutation(N) if shuffled else np.arange(N)
            k = 0
        idx = order[k:k + B]; k += B
        g = np.zeros(K)
        for c in range(K):
            m = labels[idx] == c
            if m.any():
                g[c] = 2.0 * (w[c] - y[idx][m].mean()) * m.mean()
        w = w - 0.45 * g
        losses.append(float(((w[labels] - y) ** 2).mean()))
    return np.array(losses)


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    # ---- (a) what one batch contains ---------------------------------
    ax = axes[0]
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(K), N // K)
    cols = plt.get_cmap(S.CAT)(np.linspace(0.06, 0.80, K))
    rows = [("consecutive block", np.arange(120, 120 + B)),
            ("random draw", rng.choice(N, size=B, replace=False))]
    for r, (lab, idx) in enumerate(rows):
        for j, i in enumerate(idx):
            ax.add_patch(plt.Rectangle((j, -r), 0.86, 0.86,
                                       facecolor=cols[labels[i]],
                                       edgecolor="white", lw=1.4))
        ax.annotate(lab, xy=(-0.35, -r + 0.43), fontsize=15, ha="right",
                    va="center")
    ax.set_xlim(-4.6, B + 0.2); ax.set_ylim(-1.95, 1.05)
    ax.axis("off")
    for c in range(K):
        ax.plot([], [], "s", ms=11, color=cols[c], label="class %d" % c)
    ax.legend(loc="lower center", fontsize=13, ncol=4, frameon=False,
              handletextpad=0.3, columnspacing=0.8,
              bbox_to_anchor=(0.42, -0.06))
    ax.set_title("(a) one batch of eight, from a file sorted by class")

    # ---- (b) the training curve --------------------------------------
    ax = axes[1]
    ax.plot(_curve(False, 3), color=S.OUTC, lw=1.8, label="not shuffled")
    ax.plot(_curve(True, 3), color=S.L1, lw=1.8, label="shuffled")
    ax.set_xlabel("iteration")
    ax.set_ylabel("training error")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(b) it ends up learning one class at a time")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
