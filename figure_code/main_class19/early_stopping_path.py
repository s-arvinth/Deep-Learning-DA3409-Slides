"""early_stopping_path — the gradient-descent path on a quadratic, stopped early, against the matched weight-decay solution

Used in: Class 19 (B), slide(s): "Descent on a quadratic, drawn"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python early_stopping_path.py            -> output/early_stopping_path.pdf
         python early_stopping_path.py --out DIR  -> DIR/early_stopping_path.pdf
Ported from: Class19_Regularization_Heuristics/B_Recreated_Figures/src/figures/fig02_early_vs_decay.py

Early stopping and weight decay are doing the same thing.

On a quadratic error with Hessian eigenvalues lambda_i, both methods
return a scaled version of the unregularised minimum. They differ only
in the scaling:

    gradient descent, tau steps :  ( 1 - (1 - eta lambda_i)^tau ) w*_i
    weight decay, strength alpha:  ( lambda_i / (lambda_i + alpha) ) w*_i

Both are near 1 when lambda_i is large and near 0 when it is small, and
both cross over at the same place once alpha = 1/(eta tau) -- which is
the correspondence Bishop & Bishop (2024), Section 9.3.1 states.

`early_stopping_path`  the layout of Bishop & Bishop (2024), Fig. 9.8,
    for a quadratic of this deck's own: contours of the error with the
    axes on the eigenvectors of the Hessian, and the path gradient
    descent takes from the origin. It moves first along w2, the
    direction of HIGH curvature, which converges fastest, and only
    later along w1 towards w*. The point w-hat reached after tau steps
    is marked, and so is the weight-decay solution at the matched
    penalty alpha = 1/(eta tau), so the correspondence is visible.

(Split from fig02_early_vs_decay.py, which also wrote early_vs_decay.pdf.)
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
from models import ridge_filter, matched_alpha, gd_path_quadratic

NAME = "early_stopping_path"

# the quadratic drawn in the geometry panel
Q_LAM = (0.22, 3.0)                     # w1 flat, w2 steep
Q_WSTAR = np.array([3.0, 1.45])
Q_ETA, Q_TAU = 0.10, 20


def draw_path(ax, label_panel=None):
    """Bishop Fig. 9.8's layout: fading ellipses, arrow axes, the path."""
    path = gd_path_quadratic(Q_LAM, Q_WSTAR, Q_ETA, 400)
    what = path[Q_TAU]
    a = matched_alpha(Q_ETA, Q_TAU)
    wdec = ridge_filter(Q_LAM, a) * Q_WSTAR
    print("early stopping at tau = %d: w-hat = (%.2f, %.2f); "
          "weight decay at alpha = 1/(eta tau) = %.2f: (%.2f, %.2f)"
          % (Q_TAU, *what, a, *wdec))

    g1 = np.linspace(-0.7, 5.6, 500); g2 = np.linspace(-0.6, 3.1, 400)
    G1, G2 = np.meshgrid(g1, g2)
    E = 0.5 * (Q_LAM[0] * (G1 - Q_WSTAR[0]) ** 2
               + Q_LAM[1] * (G2 - Q_WSTAR[1]) ** 2)
    levels = 0.055 * np.arange(1, 8) ** 2
    for k, lev in enumerate(levels):
        ax.contour(G1, G2, E, levels=[lev], colors=[S.OUTC], linewidths=1.3,
                   alpha=1.0 - 0.72 * k / (len(levels) - 1))
    # the path: solid up to the stopping point, dotted beyond it
    ax.plot(path[Q_TAU:, 0], path[Q_TAU:, 1], color=S.L1, lw=1.8,
            ls=(0, (2, 3)), zorder=5)
    ax.plot(path[:Q_TAU + 1, 0], path[:Q_TAU + 1, 1], color=S.L2, lw=3.2,
            zorder=6)
    ax.annotate("", xy=path[3], xytext=path[1],
                arrowprops=dict(arrowstyle="-|>", color=S.L2, lw=2.2),
                zorder=7)
    ax.plot(*Q_WSTAR, "o", ms=9, mfc=S.OUTC, mec="white", mew=1.4, zorder=8)
    ax.annotate(r"$\mathbf{w}^\star$", xy=Q_WSTAR + np.array([0.15, 0.14]),
                color=S.OUTC, fontsize=17)
    ax.plot(*what, "o", ms=9, mfc=S.L2, mec="white", mew=1.4, zorder=9)
    ax.annotate(r"$\hat{\mathbf{w}}$: stop at $\tau = %d$" % Q_TAU,
                xy=what, xytext=(1.55, 2.62), color=S.L2, fontsize=14,
                arrowprops=dict(arrowstyle="-", color=S.L2, lw=0.9),
                zorder=10)
    ax.plot(*wdec, "s", ms=9, mfc="white", mec=S.ACC, mew=2.0, zorder=9)
    ax.annotate(r"weight decay at $\alpha = 1/\eta\tau$",
                xy=wdec, xytext=(1.55, -0.42), color=S.ACC, fontsize=14,
                arrowprops=dict(arrowstyle="-", color=S.ACC, lw=0.9),
                zorder=10)
    ax.annotate("solid: the first %d steps\ndotted: the rest of the run"
                % Q_TAU, xy=(5.5, -0.42), color=S.GREY, fontsize=12,
                ha="right")
    ax.annotate(r"$E(\mathbf{w})$", xy=(4.75, 2.55), color=S.OUTC, fontsize=17)
    ax.annotate("", xy=(5.6, 0), xytext=(-0.7, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.annotate("", xy=(0, 3.1), xytext=(0, -0.6),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    ax.annotate("$w_1$", xy=(5.2, 0.14), fontsize=17)
    ax.annotate("$w_2$", xy=(0.14, 2.85), fontsize=17)
    ax.set_xlim(-0.7, 5.6); ax.set_ylim(-0.6, 3.1)
    ax.set_aspect("equal"); ax.axis("off")
    if label_panel:
        ax.set_title(label_panel)


def build_path():
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    draw_path(ax)
    fig.tight_layout()
    S.save(fig, NAME)


def build():
    S.use()
    build_path()


if __name__ == "__main__":
    build()
