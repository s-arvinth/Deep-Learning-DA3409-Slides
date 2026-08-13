"""The three kinds of stationary point, read off the Hessian spectrum.

Bishop & Bishop (2024), Section 7.1.1: substituting the eigen-expansion
into the quadratic model gives

    E(w) = E(w*) + 1/2 sum_i lambda_i alpha_i^2      (Eq. 7.11)

so the sign pattern of the eigenvalues alone decides the type.  Each
panel here is the same quadratic with a different sign pattern.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "stationary_types"

CASES = [
    ((1.0, 3.0),   "minimum",      r"$\lambda_1,\lambda_2 > 0$"),
    ((-1.0, -3.0), "maximum",      r"$\lambda_1,\lambda_2 < 0$"),
    ((-1.6, 3.0),  "saddle point", r"$\lambda_1 < 0 < \lambda_2$"),
]


def build():
    S.use()
    g = np.linspace(-1.6, 1.6, 140)
    A1, A2 = np.meshgrid(g, g)

    fig = plt.figure(figsize=(12.6, 4.5))
    for k, (lam, title, sig) in enumerate(CASES):
        Z = 0.5 * (lam[0] * A1 ** 2 + lam[1] * A2 ** 2)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot_surface(A1, A2, Z, cmap=S.SEQ, linewidth=0,
                        rstride=3, cstride=3, alpha=0.95, rasterized=True)
        ax.plot([0], [0], [0], "o", ms=7, mfc="white", mec=S.OUTC, mew=2.0)
        ax.view_init(elev=26, azim=-56)
        ax.set_xlabel(r"$\alpha_1$", labelpad=-6)
        ax.set_ylabel(r"$\alpha_2$", labelpad=-6)
        S.tidy3d(ax, ticks=(-1, 0, 1), labelsize=11)
        ax.set_zticks([])
        ax.set_title(f"{title}\n{sig}", fontsize=16, pad=-2)

    fig.tight_layout(w_pad=0.4)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
