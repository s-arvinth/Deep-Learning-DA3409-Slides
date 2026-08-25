"""Five learning-rate schedules on one pair of axes.

The three algebraic forms are Bishop's linear, power-law and
exponential decay. Cosine annealing and a linear warm-up followed by
cosine annealing are the two shapes most used in practice.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (sched_linear, sched_power, sched_exponential,
                           sched_cosine, sched_warmup_cosine)

NAME = "lr_schedules"

T = 1000
ETA0 = 0.10


def build():
    S.use()
    t = np.arange(0, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    curves = [
        ("linear", sched_linear(t, ETA0, 0.01 * ETA0, 800)),
        ("power law", sched_power(t, ETA0, 120.0, 1.0)),
        ("exponential", sched_exponential(t, ETA0, 200.0, 0.5)),
        ("cosine", sched_cosine(t, ETA0, T)),
        ("warm-up + cosine", sched_warmup_cosine(t, ETA0, T, 100)),
    ]
    cols = list(plt.get_cmap(S.CAT)(np.linspace(0.05, 0.72, 4))) + [S.GREY]

    ax = axes[0]
    for (lab, y), col in zip(curves, cols):
        ax.plot(t, y, color=col, lw=2.3, label=lab)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"learning rate $\eta^{(\tau)}$")
    ax.set_ylim(0, ETA0 * 1.12)
    ax.legend(loc="upper right", fontsize=12.5)
    ax.set_title("(a) the shapes")
    S.square(ax)

    ax = axes[1]
    for (lab, y), col in zip(curves, cols):
        ax.semilogy(t, np.maximum(y, 1e-6), color=col, lw=2.3)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\eta^{(\tau)}$   (log scale)")
    ax.set_ylim(1e-4, 0.2)
    ax.set_title("(b) the same, on a log axis")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
