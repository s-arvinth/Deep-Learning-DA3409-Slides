"""Contours of the local quadratic model are axis-aligned ellipses.

Recreates Bishop & Bishop (2024), Fig. 7.2.  In the eigen-coordinates of
the Hessian the contours of

    E(w) = E(w*) + 1/2 sum_i lambda_i alpha_i^2

are ellipses centred on w*, with semi-axes along the eigenvectors u_i and
lengths proportional to lambda_i^(-1/2).  The second panel shows what
happens to that picture as the two eigenvalues separate: the ellipse
becomes the long valley that the whole of Class 14 is about.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quadratic_E

NAME = "quadratic_ellipses"
ROT = np.deg2rad(32.0)


def build():
    S.use()
    g = np.linspace(-2.6, 2.6, 320)
    W1, W2 = np.meshgrid(g, g)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))

    # ---- (a) one quadratic, with its eigenvector axes labelled --------
    ax = axes[0]
    lam = (1.0, 4.0)
    Z = quadratic_E(W1, W2, lam=lam, rot=ROT)
    ax.contour(W1, W2, Z, levels=np.array([0.25, 1.0, 2.25, 4.0]),
               colors=[S.L1], linewidths=1.5)
    c, s = np.cos(ROT), np.sin(ROT)
    u1 = np.array([c, s])          # eigenvector for lambda_1
    u2 = np.array([-s, c])         # eigenvector for lambda_2
    for u, lm, lab, col in ((u1, lam[0], r"$\mathbf{u}_1$", S.OUTC),
                            (u2, lam[1], r"$\mathbf{u}_2$", S.L2)):
        L = 2.0 * lm ** -0.5
        ax.annotate("", xy=tuple(L * u), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
        ax.annotate(lab, xy=tuple(1.20 * L * u), color=col, fontsize=17,
                    ha="center", va="center")
        ax.annotate(r"$\lambda^{-1/2}$", xy=tuple(0.55 * L * u + 0.30 * np.array([-u[1], u[0]])),
                    color=col, fontsize=14, ha="center", va="center")
    ax.plot(0, 0, "o", ms=8, mfc="white", mec=S.OUTC, mew=2.0, zorder=5)
    ax.annotate(r"$\mathbf{w}^\star$", xy=(0.14, -0.34), fontsize=16, color=S.OUTC)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title(r"axes are the eigenvectors of $H$")
    S.square(ax)

    # ---- (b) the same picture as the eigenvalues separate -------------
    ax = axes[1]
    ratios = [1.0, 4.0, 16.0, 64.0]
    cols = plt.get_cmap(S.CAT)(np.linspace(0.08, 0.80, len(ratios)))
    for r, col in zip(ratios, cols):
        Z = quadratic_E(W1, W2, lam=(1.0, r), rot=0.0)
        ax.contour(W1, W2, Z, levels=[1.0], colors=[col], linewidths=2.0)
        ax.plot([], [], color=col, lw=2.0,
                label=r"$\lambda_2/\lambda_1 = %d$" % int(r))
    ax.plot(0, 0, "o", ms=7, mfc="white", mec=S.OUTC, mew=2.0, zorder=5)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("a wider spectrum is a narrower valley")
    ax.legend(loc="upper right", fontsize=13)
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
