"""A fixed learning rate stops improving; a decaying one does not.

With a noisy gradient and a fixed eta the iterate settles into a ball
around the minimum whose radius is set by eta, and the error stops
falling. Halving eta halves the floor but does not remove it. Letting
eta go to zero at the right speed removes it altogether.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quad_E, quad_grad

NAME = "noise_floor"

LAM = (1.0, 3.0)
START = np.array([-2.0, 1.6])
SIGMA = 1.2
T = 4000


def _run(eta_of_t, seed=0):
    rng = np.random.default_rng(seed)
    w = START.copy()
    E = np.empty(T)
    for t in range(T):
        E[t] = quad_E(w[0], w[1], lam=LAM)
        g = quad_grad(w, lam=LAM) + SIGMA * rng.standard_normal(2)
        w = w - eta_of_t(t) * g
    return E


def _smooth(x, k=41):
    return np.array([np.median(x[max(0, i - k):i + 1])
                     for i in range(len(x))])


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    for eta, col in ((0.08, S.OUTC), (0.04, S.ACC), (0.01, S.L2)):
        E = _smooth(_run(lambda t, e=eta: e))
        ax.loglog(np.arange(1, T + 1), E, color=col, lw=2.2,
                  label=r"fixed $\eta = %.2f$" % eta)
        ax.axhline(E[-1], color=col, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$E(\mathbf{w})$   (running median)")
    ax.set_ylim(1e-4, 30)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title(r"(a) a fixed $\eta$ has a floor")
    S.square(ax)

    ax = axes[1]
    E = _smooth(_run(lambda t: 0.08))
    ax.loglog(np.arange(1, T + 1), E, color=S.GREY, lw=2.2,
              label=r"fixed $\eta = 0.08$")
    E = _smooth(_run(lambda t: 0.08 / (1.0 + t / 60.0)))
    ax.loglog(np.arange(1, T + 1), E, color=S.OUTC, lw=2.2,
              label=r"$\eta^{(\tau)} \propto 1/\tau$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$E(\mathbf{w})$   (running median)")
    ax.set_ylim(1e-4, 30)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_title(r"(b) letting $\eta \to 0$ removes it")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
