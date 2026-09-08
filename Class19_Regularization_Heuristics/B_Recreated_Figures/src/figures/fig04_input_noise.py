"""Training on noisy inputs smooths the fit.

The layout of Prince (2023), Fig. 9.10, for the bump model of this deck.
At every step of SGD, fresh Gaussian noise of standard deviation
sigma_x is added to the INPUTS of the batch before the features are
computed; the model is nonlinear in x, so this is the real procedure
and not the linear shortcut.  Three noise levels, one panel each; the
small dots are ten samples of the perturbed data, exactly as Prince
draws them.  More noise gives a smoother fit.

The exact statement behind it -- for a model linear in its inputs, noise
of variance sigma_x^2 IS a quadratic penalty of strength sigma_x^2 --
is checked before the figure is drawn: a linear model is trained by
descent with fresh input noise at five noise levels and compared with
the ridge solution at lambda = sigma_x^2.  The build asserts that the
two differ by less than 2% of the size of the solution at every level,
and prints the numbers.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (truth, sample, features, linear_problem,
                           ridge_solution, train_with_input_noise,
                           sgd_input_noise)

NAME = "input_noise"

VARS = np.array([0.02, 0.05, 0.12, 0.25, 0.50])
GRID = np.linspace(0.0, 1.0, 500)
N1D, K = 26, 16
SIGMAS = (0.0, 0.02, 0.05)
N_SHOW = 10


def _linear_check():
    rng = np.random.default_rng(0)
    X, Y, _ = linear_problem(400, 8, rng)
    worst = 0.0
    for v in VARS:
        wn = train_with_input_noise(X, Y, v, rng)
        wr = ridge_solution(X, Y, v)
        rel = np.linalg.norm(wn - wr) / np.linalg.norm(wr)
        worst = max(worst, rel)
        print("sigma_x^2 = %.2f: |w_noise - w_ridge| / |w_ridge| = %.4f"
              % (v, rel))
    assert worst < 0.02
    return worst


def build():
    S.use()
    worst = _linear_check()

    rng = np.random.default_rng(7)
    x, y = sample(N1D, rng)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9))
    for ax, sig, tag in zip(axes, SIGMAS, "abc"):
        w, cen = sgd_input_noise(x, y, K, sig, np.random.default_rng(1))
        Pg, _ = features(GRID, K, centres=cen)
        f = Pg @ w
        rough = np.mean(np.abs(np.diff(f, 2))) * 1e3
        print("sigma_x = %.2f: roughness %.2f" % (sig, rough))

        ax.plot(GRID, truth(GRID), color="black", lw=2.2, zorder=3,
                label=r"$h(x)$")
        ax.plot(GRID, f, color=S.L2, lw=2.6, zorder=4, label="fit")
        if sig > 0:
            prng = np.random.default_rng(3)
            for _ in range(N_SHOW):
                ax.plot(x + sig * prng.normal(size=len(x)), y, ".",
                        ms=3.2, color=S.OUTC, alpha=0.45, zorder=5)
        ax.plot(x, y, "o", ms=6.5, mfc=S.OUTC, mec="white", mew=1.0,
                zorder=6, label="data")
        ax.set_xlim(0, 1); ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("input, $x$"); ax.set_ylabel("output, $y$")
        ax.set_title(r"(%s) $\sigma_x = %.2f$" % (tag, sig))
        ax.set_box_aspect(1.0)
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_size(13); ax.yaxis.label.set_size(13)
        ax.title.set_size(14)
    axes[0].legend(loc="lower left", fontsize=10.5)
    fig.tight_layout(w_pad=0.9)
    S.save(fig, NAME)
    return worst


if __name__ == "__main__":
    build()
