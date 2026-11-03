"""basis_functions — Three classical families of FIXED basis functions phi_j(x)

Used in: Class 5 (B), slide(s): "Beyond lines: fixed basis functions"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python basis_functions.py            -> output/basis_functions.pdf
Ported from: Class05_Shallow_NN_Regression/B_Recreated_Figures/src/figures/fig03_basis_functions.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common

NAME = "basis_functions"


def build():
    common.use()
    x = np.linspace(-1, 1, 800)
    cmap = plt.get_cmap(common.CAT)
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
        a.set_title(a.get_title(), fontsize=16.2)
        common.square(a)
    fig.tight_layout(w_pad=2.2)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
