"""The double-descent curve, laid out as the ResNet18 experiment is.

Following the geometry of Bishop & Bishop (2024), Fig. 9.9: one panel,
training and test error on the same axes, the critical regime shaded,
the interpolation threshold marked with an arrow, and the three regimes
named across the top. The data are the random-feature experiment of
fig04, re-drawn in this layout so the two figures can be read the same
way.

Left of the shaded band the test error follows the classical
bias-variance trade-off and has a minimum. Inside the band the model has
just enough capacity to fit the training data and the test error is at
its worst. Right of the band the training error is zero and the test
error falls again, to a value below the classical minimum.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from figures.fig04_double_descent import _run, PS, N

NAME = "double_descent_regimes"


def build():
    S.use()
    train, test = _run()
    lo, hi = PS < N, PS > N
    c_i = int(np.argmin(np.where(lo, test, np.inf)))
    m_i = int(np.argmin(np.where(hi, test, np.inf)))

    fig, ax = plt.subplots(1, 1, figsize=(11.0, 4.9))
    band = (0.62 * N, 1.55 * N)
    ax.axvspan(*band, color=S.OUTC, alpha=0.12, lw=0)

    ax.plot(PS, np.maximum(test, 1e-3), "-", color=S.OUTC, lw=2.8,
            label="test error")
    ax.plot(PS, np.maximum(train, 2e-3), "--", color=S.L2, lw=2.4,
            label="training error")
    ax.axvline(N, color=S.GREY, lw=1.4, ls=(0, (4, 3)))

    ax.plot(PS[c_i], test[c_i], "*", ms=17, mfc="white", mec=S.L1, mew=1.9,
            zorder=6)
    ax.plot(PS[m_i], test[m_i], "*", ms=17, mfc="white", mec=S.L1, mew=1.9,
            zorder=6)
    ax.annotate("classical minimum\n$%.3f$" % test[c_i],
                xy=(PS[c_i], test[c_i]), xytext=(PS[c_i] * 0.42, test[c_i] * 0.11),
                color=S.L1, fontsize=12.5, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=S.L1, lw=1.3))
    ax.annotate("modern minimum\n$%.3f$" % test[m_i],
                xy=(PS[m_i], test[m_i]), xytext=(PS[m_i] * 0.42, test[m_i] * 0.11),
                color=S.L1, fontsize=12.5, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=S.L1, lw=1.3))
    ax.annotate("interpolation\nthreshold", xy=(N, 2.5), xytext=(N * 4.2, 6.0),
                color=S.GREY, fontsize=12.5, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=S.GREY, lw=1.3))

    for x, lab in ((0.16, "classical regime:\nbias–variance trade-off"),
                   (0.505, "critical\nregime"),
                   (0.83, "modern regime:\nlarger model is better")):
        ax.annotate(lab, xy=(x, 0.975), xycoords="axes fraction",
                    ha="center", va="top", fontsize=12.5, color=S.GREY)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("capacity $p$   (features)")
    ax.set_ylabel("error")
    ax.set_xlim(PS[0] * 0.85, PS[-1] * 1.2)
    ax.set_ylim(1e-3, 3e4)
    ax.legend(loc="center left", fontsize=12.5)
    ax.set_box_aspect(0.42)
    fig.tight_layout()
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
