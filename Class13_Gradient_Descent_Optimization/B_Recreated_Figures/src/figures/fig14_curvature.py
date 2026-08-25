"""What the largest eigenvalue of the Hessian means, concretely.

Panel (a): a quadratic bowl whose two eigen-directions have very
different curvature, with the two principal directions drawn on the
contours.

Panel (b): the one-dimensional slice of E along each of those two
directions. The slice along u_max is the steep one; its curvature is
lambda_max. The slice along u_min is the shallow one; its curvature is
lambda_min. So lambda_max is not an abstraction -- it is the curvature
of the sharpest cross-section through the point.

Panel (c): the same two numbers as the semi-axes of one contour
ellipse, which go as lambda^(-1/2): the steep direction is the SHORT
axis.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import quadratic_E

NAME = "curvature"

LMIN, LMAX = 0.6, 6.0
ROT = np.deg2rad(30.0)


def build():
    S.use()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))

    c, s = np.cos(ROT), np.sin(ROT)
    umin = np.array([c, s])       # eigenvector of the small eigenvalue
    umax = np.array([-s, c])      # eigenvector of the large eigenvalue

    # ---- (a) the bowl, seen from above --------------------------------
    ax = axes[0]
    g = np.linspace(-2.6, 2.6, 300)
    G1, G2 = np.meshgrid(g, g)
    Z = quadratic_E(G1, G2, lam=(LMIN, LMAX), rot=ROT)
    ax.contourf(G1, G2, Z, levels=22, cmap=S.SEQ, alpha=0.92)
    ax.contour(G1, G2, Z, levels=22, colors="white", linewidths=0.4,
               alpha=0.55)
    for u, col, lab in ((umin, "#F2F2F2", r"$\mathbf{u}_{min}$"),
                        (umax, S.OUTC, r"$\mathbf{u}_{max}$")):
        ax.annotate("", xy=tuple(2.15 * u), xytext=tuple(-2.15 * u),
                    arrowprops=dict(arrowstyle="<|-|>", color=col, lw=2.2))
        ax.annotate(lab, xy=tuple(2.42 * u), color=col, fontsize=17,
                    ha="center", va="center")
    ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8, zorder=6)
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.7, 2.7)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("(a) the two principal directions")
    S.square(ax)

    # ---- (b) the slice along each direction ---------------------------
    ax = axes[1]
    t = np.linspace(-2.2, 2.2, 400)
    ax.plot(t, 0.5 * LMIN * t ** 2, color=S.L2, lw=2.6,
            label=r"along $u_{min}$:  $\frac{1}{2}\lambda_{min}t^2$")
    ax.plot(t, 0.5 * LMAX * t ** 2, color=S.OUTC, lw=2.6,
            label=r"along $u_{max}$:  $\frac{1}{2}\lambda_{max}t^2$")
    ax.annotate(r"$\lambda_{max} = %.1f$" % LMAX, xy=(1.05, 0.5 * LMAX),
                xytext=(0.30, 6.2), fontsize=15, color=S.OUTC)
    ax.annotate(r"$\lambda_{min} = %.1f$" % LMIN, xy=(1.9, 0.5 * LMIN * 3.6),
                xytext=(0.55, 1.55), fontsize=15, color=S.L2)
    ax.set_xlabel(r"distance $t$ from $\mathbf{w}^\star$")
    ax.set_ylabel(r"$E(\mathbf{w}^\star + t\,\mathbf{u}) - E(\mathbf{w}^\star)$")
    ax.set_ylim(-0.3, 8.0)
    ax.legend(loc="upper center", fontsize=13)
    ax.set_title("(b) the eigenvalue is a curvature")
    S.square(ax)

    # ---- (c) the same numbers as the axes of one ellipse --------------
    ax = axes[2]
    Z1 = quadratic_E(G1, G2, lam=(LMIN, LMAX), rot=ROT)
    ax.contour(G1, G2, Z1, levels=[1.0], colors=[S.L1], linewidths=2.6)
    for u, lam, col, lab in ((umin, LMIN, S.L2, r"$\lambda_{min}^{-1/2}$"),
                             (umax, LMAX, S.OUTC, r"$\lambda_{max}^{-1/2}$")):
        Lh = np.sqrt(2.0 / lam)
        ax.annotate("", xy=tuple(Lh * u), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        off = 0.62 if lam == LMAX else 0.55
        ax.annotate(lab, xy=tuple(1.30 * Lh * u
                                  + off * np.array([-u[1], u[0]])),
                    color=col, fontsize=15, ha="center", va="center")
    ax.plot(0, 0, "*", ms=14, mfc="white", mec=S.OUTC, mew=1.8, zorder=6)
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.7, 2.7)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("(c) steep direction = short axis")
    S.square(ax)

    fig.tight_layout(w_pad=1.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
