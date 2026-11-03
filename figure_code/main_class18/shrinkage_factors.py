"""shrinkage_factors — each direction's shrinkage factor lambda_i/(lambda_i + alpha) against alpha

Used in: Class 18 (B), slide(s): "The flat directions go first"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python shrinkage_factors.py            -> output/shrinkage_factors.pdf
         python shrinkage_factors.py --out DIR  -> DIR/shrinkage_factors.pdf
Ported from: Class18_Regularization_Explicit_Implicit/B_Recreated_Figures/src/figures/fig02_weight_decay_geometry.py

`shrinkage_factors`  each direction's factor lambda_i/(lambda_i + alpha)
    against alpha, with their sum, the effective number of parameters.

LAM and ALPHA are the values of the weight_decay_geometry figure.
(Split from fig02_weight_decay_geometry.py, which also wrote weight_decay_geometry.pdf.)
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S

NAME = "shrinkage_factors"

LAM = (0.30, 3.20)                 # eigenvalues: shallow along u1, steep along u2
ALPHA = 1.10


def build_factors():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    a = np.logspace(-2.2, 2.0, 400)
    for lam, col, lab in ((LAM[0], S.OUTC, r"$\lambda_1 = %.2f$  (shallow)" % LAM[0]),
                          (LAM[1], S.L2, r"$\lambda_2 = %.2f$  (steep)" % LAM[1])):
        ax.semilogx(a, lam / (lam + a), color=col, lw=2.6, label=lab)
        ax.plot(lam, 0.5, "o", ms=7, mfc="white", mec=col, mew=1.9, zorder=6)
    ax.semilogx(a, LAM[0] / (LAM[0] + a) + LAM[1] / (LAM[1] + a),
                color=S.L1, lw=2.2, ls=(0, (5, 2.5)),
                label="effective parameters")
    ax.axvline(ALPHA, color=S.GREY, lw=1.2, ls=(0, (3, 3)))
    ax.annotate(r"$\alpha$ used in the contour figure", xy=(ALPHA * 1.15, 1.88),
                color=S.GREY, fontsize=13)
    ax.annotate("half shrunk at $\\alpha = \\lambda_i$", xy=(LAM[1] * 1.35, 0.42),
                color=S.GREY, fontsize=12.5)
    ax.set_xlabel(r"penalty strength $\alpha$")
    ax.set_ylabel(r"$\lambda_i / (\lambda_i + \alpha)$")
    ax.set_ylim(0, 2.15)
    ax.legend(loc="lower left", fontsize=12.5)
    S.square(ax)
    fig.tight_layout()
    S.save(fig, NAME)


def build():
    S.use()
    build_factors()


if __name__ == "__main__":
    build()
