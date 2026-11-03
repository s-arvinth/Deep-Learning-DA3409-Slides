"""qnorm_contours — level sets of sum |w_j|^q for q = 0.5, 1, 2, 4

Used in: Class 18 (B), slide(s): "Other shapes of penalty"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python qnorm_contours.py            -> output/qnorm_contours.pdf
         python qnorm_contours.py --out DIR  -> DIR/qnorm_contours.pdf
Ported from: Class18_Regularization_Explicit_Implicit/B_Recreated_Figures/src/figures/fig03_lasso_vs_ridge.py

Why q = 1 sets coefficients to zero and q = 2 does not.

Following the geometry of Bishop & Bishop (2024), Figs. 9.5
and 9.6.

`qnorm_contours`  the level sets of sum |w_j|^q for four values of q.
    Only q <= 1 gives a region with corners on the axes, and only q >= 1
    gives a convex one. q = 1 is the single value with both.

(Split from fig03_lasso_vs_ridge.py, which also wrote lasso_vs_ridge.pdf.)
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S

NAME = "qnorm_contours"

QS = (0.5, 1.0, 2.0, 4.0)


def _arrow_axes(ax, lim, lab=True):
    """Axes drawn as arrows through the origin, as in Bishop's figures."""
    ax.annotate("", xy=(lim, 0), xytext=(-lim, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.annotate("", xy=(0, lim), xytext=(0, -lim),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    if lab:
        ax.annotate("$w_1$", xy=(lim - 0.36, -0.30), fontsize=16)
        ax.annotate("$w_2$", xy=(0.08, lim - 0.28), fontsize=16)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def build_contours():
    """Level sets of sum |w_j|^q for four q, the layout of Bishop Fig. 9.5:
    a family of nested level sets that fade outwards, on arrow axes."""
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.3))
    levels = np.linspace(0.10, 1.0, 8)
    for ax, q in zip(axes, QS):
        t = np.linspace(0, 2 * np.pi, 3000)
        c, s = np.cos(t), np.sin(t)
        r = (np.abs(c) ** q + np.abs(s) ** q) ** (-1.0 / q)
        for k, lev in enumerate(levels):
            a = 0.30 + 0.70 * (k + 1) / len(levels)
            ax.plot(lev ** (1.0 / q) * r * c, lev ** (1.0 / q) * r * s,
                    color=S.OUTC, lw=1.5 if k < len(levels) - 1 else 2.3,
                    alpha=a)
        _arrow_axes(ax, 1.30)
        ax.set_title(r"$q = %g$" % q, y=-0.13)
    fig.tight_layout(w_pad=1.0)
    S.save(fig, NAME)


def build():
    S.use()
    build_contours()


if __name__ == "__main__":
    build()
