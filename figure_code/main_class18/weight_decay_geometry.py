"""weight_decay_geometry — error, penalty and their sum as contours; how far weight decay moves w along each eigen-direction

Used in: Class 18 (B), slide(s): "The same statement, drawn"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python weight_decay_geometry.py            -> output/weight_decay_geometry.pdf
         python weight_decay_geometry.py --out DIR  -> DIR/weight_decay_geometry.pdf
Ported from: Class18_Regularization_Explicit_Implicit/B_Recreated_Figures/src/figures/fig02_weight_decay_geometry.py

What weight decay does, and to which directions.

the layout of Bishop & Bishop (2024), Fig. 9.3,
    for a quadratic error of this deck's own.  Contours of the error
    (maroon) centred on w*, of the penalty (teal) centred on the origin,
    and of their sum (indigo) centred on w-hat.  The eigenvectors u1, u2
    of the Hessian are drawn at w*: the error is elongated along u1, so
    it hardly objects to movement that way, and the regularised minimum
    slides a long way along u1 and only a little along u2.

    The point w-hat is computed as (H + alpha I)^{-1} H w*, and the
    build asserts that it equals the eigenbasis shrinkage
    lambda_i/(lambda_i + alpha) of Theorem 3.

(Split from fig02_weight_decay_geometry.py, which also wrote shrinkage_factors.pdf.)
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
from models import rotated_hessian, quad_E_rot, ridge_solution_rot

NAME = "weight_decay_geometry"

LAM = (0.30, 3.20)                 # eigenvalues: shallow along u1, steep along u2
THETA = np.deg2rad(14.0)           # orientation of u1 relative to w1
WSTAR = np.array([2.70, 1.35])
ALPHA = 1.10


def _fading_contours(ax, Z, levels, col, lw=1.35):
    """Bishop draws many level sets that fade outwards; so do we."""
    n = len(levels)
    for k, lev in enumerate(levels):
        a = 1.0 - 0.80 * k / max(n - 1, 1)
        ax.contour(*Z, levels=[lev], colors=[col], linewidths=lw, alpha=a)


def build_geometry():
    H, R = rotated_hessian(LAM, THETA)
    u1, u2 = R[:, 0], R[:, 1]
    what = ridge_solution_rot(H, WSTAR, ALPHA)

    # ---- the check against Theorem 3 ---------------------------------
    wst_e = R.T @ WSTAR
    what_e = np.array(LAM) / (np.array(LAM) + ALPHA) * wst_e
    assert np.allclose(R @ what_e, what, atol=1e-12)
    move = R.T @ (WSTAR - what)
    print("w* = (%.2f, %.2f), w-hat = (%.2f, %.2f)" % (*WSTAR, *what))
    print("moved %.2f along u1 (shallow) and %.2f along u2 (steep)" % tuple(move))

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    g1 = np.linspace(-1.9, 6.6, 760)
    g2 = np.linspace(-1.5, 3.3, 460)
    G1, G2 = np.meshgrid(g1, g2)
    E = quad_E_rot(G1, G2, H, WSTAR)
    P = 0.5 * (G1 ** 2 + G2 ** 2)
    T = E + ALPHA * P

    _fading_contours(ax, (G1, G2, E), 0.05 * np.arange(1, 7) ** 2, S.OUTC)
    _fading_contours(ax, (G1, G2, P), 0.036 * np.arange(1, 6) ** 2, S.L2)
    Tmin = float(quad_E_rot(what[0], what[1], H, WSTAR)
                 + ALPHA * 0.5 * what @ what)
    _fading_contours(ax, (G1, G2, T), Tmin + 0.12 * np.arange(1, 6) ** 2, S.L1)

    # ---- the eigen-axes at w* ----------------------------------------
    for u, lab, off in ((u1, r"$\mathbf{u}_1$", (0.05, -0.28)),
                        (u2, r"$\mathbf{u}_2$", (0.12, 0.0))):
        ax.annotate("", xy=WSTAR + 1.5 * u, xytext=WSTAR,
                    arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.4),
                    zorder=8)
        ax.annotate(lab, xy=WSTAR + 1.5 * u + np.array(off), color=S.OUTC,
                    fontsize=15)

    # ---- the two minima and the move ---------------------------------
    ax.plot(*WSTAR, "o", ms=8, mfc=S.OUTC, mec="white", mew=1.4, zorder=9)
    ax.annotate(r"$\mathbf{w}^\star$", xy=WSTAR + np.array([-0.42, 0.10]),
                color=S.OUTC, fontsize=16)
    ax.plot(*what, "o", ms=8, mfc=S.L1, mec="white", mew=1.4, zorder=9)
    ax.annotate(r"$\hat{\mathbf{w}}$", xy=what + np.array([-0.40, 0.10]),
                color=S.L1, fontsize=16)
    ax.annotate("", xy=tuple(what), xytext=tuple(WSTAR),
                arrowprops=dict(arrowstyle="-|>", color=S.GREY, lw=1.4,
                                ls=(0, (4, 2.5))), zorder=7)

    # ---- the three equations, on the figure as in Bishop -------------
    ax.annotate(r"$E(\mathbf{w})$", xy=(4.6, 2.75), color=S.OUTC, fontsize=17)
    ax.annotate(r"$w_1^2 + w_2^2$", xy=(0.25, -1.40), color=S.L2, fontsize=17)
    ax.annotate(r"$E(\mathbf{w}) + \frac{\alpha}{2}\,(w_1^2 + w_2^2)$",
                xy=(-1.85, 2.05), color=S.L1, fontsize=17)

    # ---- axes drawn as arrows through the origin ---------------------
    ax.annotate("", xy=(6.55, 0), xytext=(-1.9, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
    ax.annotate("", xy=(0, 3.28), xytext=(0, -1.5),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
    ax.annotate("$w_1$", xy=(6.2, -0.34), fontsize=17)
    ax.annotate("$w_2$", xy=(0.12, 2.98), fontsize=17)
    ax.set_xlim(-1.9, 6.6); ax.set_ylim(-1.5, 3.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    S.save(fig, NAME)


def build():
    S.use()
    build_geometry()


if __name__ == "__main__":
    build()
