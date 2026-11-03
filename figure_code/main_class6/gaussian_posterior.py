"""gaussian_posterior — Gaussian class-conditionals with shared variance give a sigmoid posterior

Used in: Class 6 (B), slide(s): "The sigmoid, seen in the data"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python gaussian_posterior.py            -> output/gaussian_posterior.pdf
Ported from: Class06_Shallow_NN_Classification/B_Recreated_Figures/src/figures/fig03_gaussian_posterior.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import L1, L2, OUTC, ACC, GREY, sigmoid

NAME = "gaussian_posterior"


def build():
    common.use()
    x = np.linspace(-6, 6, 900)
    m1, m2, s = -1.6, 1.6, 1.15

    def g(mu):
        return np.exp(-(x - mu) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))

    p1, p2 = g(m1), g(m2)
    post = p2 / (p1 + p2)
    a = (m2 - m1) / s ** 2 * x - (m2 ** 2 - m1 ** 2) / (2 * s ** 2)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))
    ax[0].plot(x, p1, color=L1, lw=2.4, label=r"$p(x\mid\mathcal{C}_1)$")
    ax[0].plot(x, p2, color=OUTC, lw=2.4, label=r"$p(x\mid\mathcal{C}_2)$")
    ax[0].fill_between(x, 0, np.minimum(p1, p2), color=GREY, alpha=.25, lw=0)
    ax[0].legend(loc="upper left", fontsize=14.3)
    ax[0].set(xlabel=r"$x$", ylabel="density")
    ax[0].set_title(r"(a)  shared variance $\sigma^2$", fontsize=16.2)

    ax[1].plot(x, post, color=L2, lw=2.8,
               label=r"$p(\mathcal{C}_2\mid x)$")
    ax[1].plot(x, sigmoid(a), color=ACC, lw=1.6, ls="--",
               label=r"$\sigma(w x + w_0)$")
    ax[1].axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax[1].axvline(0.0, color=GREY, lw=0.8, ls=":")
    ax[1].legend(loc="upper left", fontsize=14.3)
    ax[1].set(xlabel=r"$x$", ylabel="posterior probability",
              ylim=(-0.04, 1.04))
    ax[1].set_title(r"(b)  the posterior is exactly a sigmoid",
                    fontsize=16.2)
    for a_ in ax:
        common.square(a_)
    fig.tight_layout(w_pad=2.4)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
