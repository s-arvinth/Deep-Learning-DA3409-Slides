#!/usr/bin/env python3
"""Three classical families of FIXED basis functions phi_j(x)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np


NAME = "basis_functions"


def build():
    style.use()
    x = np.linspace(-1, 1, 800)
    cmap = plt.get_cmap(style.CAT)
    cols = cmap(np.linspace(0.05, 0.80, 8))

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))

    for j in range(8):                                     # polynomial
        ax[0].plot(x, x ** j, lw=1.7, color=cols[j])
    ax[0].set(xlabel=r"$x$", ylabel=r"$\phi_j(x)$", ylim=(-1.15, 1.15),
              title=r"(a)  polynomial:  $\phi_j(x)=x^{\,j}$")

    s = 0.16
    for j, mu in enumerate(np.linspace(-0.85, 0.85, 8)):    # Gaussian
        ax[1].plot(x, np.exp(-((x - mu) ** 2) / (2 * s ** 2)), lw=1.7,
                   color=cols[j])
    ax[1].set(xlabel=r"$x$", ylim=(-0.08, 1.12),
              title=r"(b)  Gaussian:  $\exp(-\frac{(x-\mu_j)^2}{2s^2})$")

    for j, mu in enumerate(np.linspace(-0.85, 0.85, 8)):    # sigmoid
        ax[2].plot(x, 1 / (1 + np.exp(-(x - mu) / 0.13)), lw=1.7,
                   color=cols[j])
    ax[2].set(xlabel=r"$x$", ylim=(-0.08, 1.12),
              title=r"(c)  sigmoid:  $\sigma((x-\mu_j)/s)$")

    for a in ax:
        a.set_title(a.get_title(), fontsize=12.5)
        style.square(a)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
