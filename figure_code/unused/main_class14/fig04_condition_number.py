"""What the condition number costs.

Panel (a): the same bowl at four values of kappa = lambda_max/lambda_min.
As kappa grows the circular contours stretch into a long narrow valley.

Panel (b): gradient descent run at its largest safe learning rate on
each of those bowls, counting the steps needed to reduce the error by
six orders of magnitude. The count grows linearly in kappa, which is
what Bishop's factor (1 - 2 lambda_min / lambda_max) predicts.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quad_E, quad_grad

NAME = "condition_number"


def _steps(kap, tol=1e-6):
    lam = (1.0, kap)
    eta = 1.0 / kap                     # just inside 2/lambda_max
    w = np.array([1.0, 1.0])
    for t in range(1, 200001):
        w = w - eta * quad_grad(w, lam=lam)
        if 0.5 * (lam[0] * w[0] ** 2 + lam[1] * w[1] ** 2) < tol:
            return t
    return np.nan


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    g = np.linspace(-2.6, 2.6, 320)
    G1, G2 = np.meshgrid(g, g)
    kaps = [1, 4, 16, 64]
    cols = plt.get_cmap(S.CAT)(np.linspace(0.06, 0.80, len(kaps)))
    for k, col in zip(kaps, cols):
        Z = quad_E(G1, G2, lam=(1.0, float(k)))
        ax.contour(G1, G2, Z, levels=[1.0], colors=[col], linewidths=2.2)
        ax.plot([], [], color=col, lw=2.2, label=r"$\kappa = %d$" % k)
    ax.plot(0, 0, "*", ms=14, mfc="white", mec=S.OUTC, mew=1.8, zorder=6)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(a) one contour, four condition numbers")
    S.square(ax)

    ax = axes[1]
    ks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    n = np.array([_steps(float(k)) for k in ks])
    ax.loglog(ks, n, "o-", color=S.L1, lw=2.2, ms=7, mfc="white", mew=1.8,
              label="measured")
    ax.loglog(ks, n[3] * ks / ks[3], "--", color=S.GREY, lw=1.8,
              label=r"proportional to $\kappa$")
    ax.set_xlabel(r"condition number $\kappa$")
    ax.set_ylabel(r"steps to reduce $E$ by $10^{6}$")
    ax.legend(loc="upper left", fontsize=13)
    ax.set_title(r"(b) the cost is linear in $\kappa$")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
