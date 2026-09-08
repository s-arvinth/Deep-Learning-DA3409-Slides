"""The same experiment at four noise levels.

Following the comparison of Prince (2023), Fig. 8.10, whose point is
that the peak is more pronounced when the labels are noisier. The
random-feature model of fig04 is re-run with the label-noise standard
deviation set to four values and everything else held fixed.

With no noise there is barely a peak: the model at the threshold has
nothing spurious to fit. As the noise grows the peak grows with it,
because at the threshold the model is forced to interpolate the noise
exactly, and the only way to do so is with an enormous parameter vector.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from figures import fig04_double_descent as F

NAME = "double_descent_noise"

SIGMAS = (0.0, 0.25, 0.55, 1.0)


def _run_with_noise(sigma, seed=1):
    old = F.SIGMA
    F.SIGMA = sigma
    try:
        return F._run(seed)
    finally:
        F.SIGMA = old


def build():
    S.use()
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 4.2))
    for ax, sig, tag in zip(axes, SIGMAS, "abcd"):
        train, test = _run_with_noise(sig)
        ax.axvline(F.N, color=S.GREY, lw=1.2, ls=(0, (4, 3)))
        ax.plot(F.PS, np.maximum(test, 1e-4), "-o", color=S.OUTC, lw=2.2,
                ms=3.4, label="test")
        ax.plot(F.PS, np.maximum(train, 1e-4), "--", color=S.L2, lw=1.9,
                label="train")
        pk = test[np.argmin(np.abs(F.PS - F.N))]
        ax.annotate("peak %.2g" % pk, xy=(0.96, 0.05),
                    xycoords="axes fraction", ha="right", fontsize=12,
                    color=S.OUTC)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1e-4, 1e4)
        ax.set_xlabel("capacity $p$")
        ax.set_title(r"(%s) $\sigma = %g$" % (tag, sig))
        S.square(ax)
    axes[0].set_ylabel("mean squared error")
    axes[0].legend(loc="upper right", fontsize=11.5)
    fig.tight_layout(w_pad=1.3)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
