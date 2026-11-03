"""pla_run — the PLA boundary over successive corrections, four snapshots

Used in: Class 3 (B), slide(s): "The algorithm running"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python pla_run.py            -> output/pla_run.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L2, GREY, scatter_PN, boundary, pla, save

NAME = "pla_run"


def build():
    # clouds close together so the algorithm needs several corrections
    r = np.random.default_rng(4)
    P = r.normal([0.9, 0.7], 0.55, (22, 2))
    N = r.normal([-0.8, -0.6], 0.55, (22, 2))
    hist = pla(P, N, np.array([2.5, 2.2, -2.6]),
               rng=np.random.default_rng(3))
    xs = np.linspace(-3.2, 3.2, 20)
    last = len(hist) - 1
    picks = sorted(set([0, max(1, last // 3), max(2, 2 * last // 3), last]))
    while len(picks) < 4:
        picks.append(last)
    picks = picks[:4]

    fig, ax = plt.subplots(1, 4, figsize=(11.4, 3.0))
    for a, k in zip(ax, picks):
        scatter_PN(a, P, N, s=30)
        # faded trail of earlier boundaries
        for j in range(k):
            boundary(a, hist[j], xs, color=GREY, lw=0.7, alpha=.28)
        boundary(a, hist[k], xs, color=L2, lw=2.0)
        if k == 0:
            ttl = "start (random $w$)"
        elif k == len(hist) - 1:
            ttl = f"converged: {k} updates"
        else:
            ttl = f"after {k} update" + ("s" if k != 1 else "")
        a.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xlabel=r"$x_1$",
              title=ttl)
        a.set_aspect("equal", "box")
    ax[0].set_ylabel(r"$x_2$")
    ax[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    return save(fig, NAME)


if __name__ == "__main__":
    build()
