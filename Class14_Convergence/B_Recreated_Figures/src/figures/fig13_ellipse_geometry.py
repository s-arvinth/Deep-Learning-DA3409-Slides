"""The condition number, read straight off a contour ellipse.

A contour of the quadratic model is the set where

    1/2 ( lambda_min alpha_min^2 + lambda_max alpha_max^2 ) = c ,

which is an ellipse whose axes lie along the eigenvectors and whose
semi-axes are sqrt(2c) / sqrt(lambda). The steeply curving direction
therefore gives the SHORT axis, and the ratio of the two semi-axes is

    a_long / a_short = sqrt(lambda_max / lambda_min) = sqrt(kappa) .

So kappa is not an abstraction: it is the elongation of the contour,
squared. Panel (a) names every part of that picture. Panels (b) and (c)
show why the elongation matters for descent: on a circle the negative
gradient points straight at the minimum, while on an elongated ellipse
it does not, and the angle it misses by grows with kappa.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "ellipse_geometry"

LMIN, LMAX = 1.0, 6.25          # kappa = 6.25, so sqrt(kappa) = 2.5
ROT = np.deg2rad(28.0)


def _ellipse(lmin, lmax, rot, c=1.0, n=400):
    """Points on 1/2 (lmin a1^2 + lmax a2^2) = c, rotated by `rot`."""
    t = np.linspace(0, 2 * np.pi, n)
    a1 = np.sqrt(2 * c / lmin) * np.cos(t)
    a2 = np.sqrt(2 * c / lmax) * np.sin(t)
    cs, sn = np.cos(rot), np.sin(rot)
    return cs * a1 - sn * a2, sn * a1 + cs * a2


def build():
    S.use()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))

    # ---- (a) one ellipse, every part named -------------------------
    ax = axes[0]
    cs, sn = np.cos(ROT), np.sin(ROT)
    umin = np.array([cs, sn])        # eigenvector of the SMALL eigenvalue
    umax = np.array([-sn, cs])       # eigenvector of the LARGE eigenvalue
    C0 = 2.0                         # the contour we label
    a_long = np.sqrt(2.0 * C0 / LMIN)    # semi-axis along u_min
    a_short = np.sqrt(2.0 * C0 / LMAX)   # semi-axis along u_max

    for c, alpha in ((C0, 1.0), (0.9 * C0, 0.35), (0.35 * C0, 0.25)):
        x, y = _ellipse(LMIN, LMAX, ROT, c)
        ax.plot(x, y, color=S.L1, lw=2.6 if c == C0 else 1.1,
                alpha=alpha, zorder=3)

    # the two semi-axes, drawn from the centre
    ax.annotate("", xy=tuple(a_long * umin), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=S.L2, lw=2.4),
                zorder=5)
    ax.annotate("", xy=tuple(a_short * umax), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=2.4),
                zorder=5)

    # labels live in the two empty corners, with a leader to the arrow tip
    ax.annotate(r"$\mathbf{u}_{max}$" "\n" r"semi-axis $\lambda_{max}^{-1/2}$",
                xy=tuple(0.96 * a_short * umax), xytext=(-2.55, 2.05),
                color=S.OUTC, fontsize=14, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=S.OUTC, lw=0.9,
                                shrinkA=4, shrinkB=3))
    ax.annotate(r"$\mathbf{u}_{min}$" "\n" r"semi-axis $\lambda_{min}^{-1/2}$",
                xy=tuple(0.97 * a_long * umin), xytext=(2.55, -2.05),
                color=S.L2, fontsize=14, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=S.L2, lw=0.9,
                                shrinkA=4, shrinkB=3))

    ax.plot(0, 0, "*", ms=16, mfc="white", mec=S.OUTC, mew=1.9, zorder=6)
    ax.annotate(r"$\mathbf{w}^\star$", xy=(0.18, -0.55), color=S.OUTC,
                fontsize=15)

    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.7, 2.7)
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_title("(a) the axes are the eigenvectors")
    S.square(ax)

    # ---- (b), (c) the gradient points at the minimum only if kappa = 1
    for ax, kap, tag in ((axes[1], 1.0, "(b)"), (axes[2], 25.0, "(c)")):
        lam = np.array([1.0, kap])
        for c, alpha, lw in ((2.0, 1.0, 2.4), (0.9, 0.45, 1.1),
                             (0.25, 0.3, 1.1)):
            x, y = _ellipse(lam[0], lam[1], 0.0, c)
            ax.plot(x, y, color=S.L1, lw=lw, alpha=alpha, zorder=3)

        w = np.array([1.55, np.sqrt(2 * 2.0 / kap) * 0.62])  # a point on it
        g = -lam * w                                          # the step -grad E
        g = 1.35 * g / np.linalg.norm(g)
        d = -w
        d = 1.35 * d / np.linalg.norm(d)

        ax.annotate("", xy=tuple(w + g), xytext=tuple(w),
                    arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=2.4),
                    zorder=6)
        ax.plot(*w, "o", ms=8, mfc="white", mec=S.OUTC, mew=2.0, zorder=7)
        ax.plot(0, 0, "*", ms=15, mfc="white", mec=S.OUTC, mew=1.8, zorder=7)

        ang = np.degrees(np.arccos(np.clip(
            g @ d / (np.linalg.norm(g) * np.linalg.norm(d)), -1, 1)))

        if ang < 1.0:
            # the two arrows coincide: label them once
            ax.annotate(r"$-\nabla E$  points at  $\mathbf{w}^\star$",
                        xy=(0.5, 0.90), xycoords="axes fraction",
                        color=S.OUTC, fontsize=14, ha="center")
        else:
            ax.annotate("", xy=tuple(w + d), xytext=tuple(w),
                        arrowprops=dict(arrowstyle="-|>", color=S.L2, lw=2.0,
                                        ls=(0, (4, 2.5))), zorder=5)
            perp = np.array([-d[1], d[0]])
            perp = perp / np.linalg.norm(perp)
            if perp[1] < 0:
                perp = -perp
            ax.annotate(r"towards $\mathbf{w}^\star$",
                        xy=tuple(w + 0.55 * d + 0.72 * perp), color=S.L2,
                        fontsize=13.5, ha="center", va="center")
            ax.annotate(r"$-\nabla E$", xy=tuple(w + 1.20 * g), color=S.OUTC,
                        fontsize=14, ha="center", va="center")

        ax.annotate(r"miss $= %.0f^\circ$" % ang, xy=(0.5, 0.045),
                    xycoords="axes fraction", color=S.GREY, fontsize=14,
                    ha="center")

        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        ax.set_title(r"%s $\kappa = %d$" % (tag, int(kap)))
        S.square(ax)

    fig.tight_layout(w_pad=1.4)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
