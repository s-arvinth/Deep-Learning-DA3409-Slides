"""The posterior over parameters, drawn.

The layout of Prince (2023), Fig. 9.11, for the bump model of this deck.
With a Gaussian prior N(0, sigma_w^2 I) on the weights and Gaussian
noise on the targets the posterior is Gaussian in closed form, so
nothing here is approximated: the samples are drawn from the exact
posterior and the band is the exact predictive standard deviation.

(a-c) Two parameter vectors drawn from the posterior under priors of
      three variances.  A tighter prior gives smaller parameters and
      smoother functions.
(d-f) The predictive mean -- the average over every parameter vector,
      weighted by the posterior -- with two standard deviations shaded.
      The tighter prior also gives a narrower band.

The norm of the posterior mean and the width of the band are printed,
so the two trends in the caption are measured rather than described.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (truth, sample, features, bayes_posterior,
                           bayes_predictive, SIGMA)

NAME = "bayesian"

PRIORS = (10.0, 0.3, 0.01)
GRID = np.linspace(0.0, 1.0, 500)
N1D, K = 26, 16


def build():
    S.use()
    rng = np.random.default_rng(7)
    x, y = sample(N1D, rng)
    P, cen = features(x, K)
    Pg, _ = features(GRID, K, centres=cen)

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.1))
    srng = np.random.default_rng(11)
    for j, pv in enumerate(PRIORS):
        m, Sg = bayes_posterior(P, y, SIGMA ** 2, pv)
        mu, sd = bayes_predictive(Pg, m, Sg, SIGMA ** 2)
        draws = srng.multivariate_normal(m, Sg, size=2)
        print("prior variance %g: |mean| = %.2f, band width %.2f to %.2f"
              % (pv, np.linalg.norm(m[1:]), 2 * sd.min(), 2 * sd.max()))

        ax = axes[0, j]
        ax.plot(GRID, truth(GRID), color="black", lw=2.2, zorder=3)
        for d, col in zip(draws, (S.L2, S.ACC)):
            ax.plot(GRID, Pg @ d, color=col, lw=2.2, zorder=4)
        ax.plot(x, y, "o", ms=6, mfc=S.OUTC, mec="white", mew=1.0, zorder=6)
        ax.set_title(r"(%s) $\sigma_w^2 = %g$" % ("abc"[j], pv))

        ax = axes[1, j]
        ax.fill_between(GRID, mu - 2 * sd, mu + 2 * sd, color=S.GREY,
                        alpha=0.28, lw=0, zorder=2)
        ax.plot(GRID, truth(GRID), color="black", lw=2.2, zorder=3)
        ax.plot(GRID, mu, color=S.L2, lw=2.4, zorder=4)
        ax.plot(x, y, "o", ms=6, mfc=S.OUTC, mec="white", mew=1.0, zorder=6)
        ax.set_title("(%s) predictive mean, $\\pm 2$ s.d." % "def"[j])

    for ax in axes.ravel():
        ax.set_xlim(0, 1); ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("input, $x$"); ax.set_ylabel("output, $y$")
        ax.set_box_aspect(0.55)
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12)
        ax.title.set_size(11.5)
    axes[0, 0].plot([], [], color="black", lw=2.2, label=r"$h(x)$")
    axes[0, 0].plot([], [], color=S.L2, lw=2.2, label="posterior samples")
    axes[0, 0].legend(loc="lower left", fontsize=9.5)
    fig.tight_layout(w_pad=0.8, h_pad=0.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
