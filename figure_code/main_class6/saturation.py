"""saturation — Why squared error stalls on a sigmoid output and cross-entropy does not

Used in: Class 6 (B), slide(s): "Why not squared error on probabilities?"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python saturation.py            -> output/saturation.pdf
Ported from: Class06_Shallow_NN_Classification/B_Recreated_Figures/src/figures/fig08_saturation.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import L2, OUTC, ACC, sigmoid, dsigmoid

NAME = "saturation"


def build():
    common.use()
    a = np.linspace(-8, 8, 900)
    yhat = sigmoid(a)
    y = 1.0                                   # true label is 1

    E_ce = -np.log(np.clip(yhat, 1e-12, 1))
    E_se = (yhat - y) ** 2
    g_ce = yhat - y
    g_se = (yhat - y) * dsigmoid(a)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.5))
    ax[0].plot(a, E_ce, color=OUTC, lw=2.6, label=r"cross-entropy")
    ax[0].plot(a, E_se, color=L2, lw=2.6, ls="--", label=r"squared error")
    ax[0].legend(loc="upper right", fontsize=14.3)
    ax[0].set(xlabel=r"pre-activation $a$", ylabel=r"$E$ for one point",
              ylim=(-0.2, 6))
    ax[0].set_title(r"(a)  the error when the true label is $y=1$",
                    fontsize=15.6)

    ax[1].plot(a, np.abs(g_ce), color=OUTC, lw=2.6,
               label=r"$|\hat y-y|$   (cross-entropy)")
    ax[1].plot(a, np.abs(g_se), color=L2, lw=2.6, ls="--",
               label=r"$|(\hat y-y)\,\sigma'(a)|$   (squared)")
    ax[1].fill_between(a, 0, 1.05, where=a < -4, color=ACC, alpha=.18, lw=0)
    ax[1].annotate("confidently\nwrong", xy=(-7.7, 0.62), fontsize=14.3,
                   color=ACC)
    ax[1].legend(loc="upper right", fontsize=12.7)
    ax[1].set(xlabel=r"pre-activation $a$",
              ylabel=r"$|\partial E/\partial a|$", ylim=(0, 1.08))
    ax[1].set_title(r"(b)  squared error stops learning; "
                    r"cross-entropy does not", fontsize=15.0)
    for a_ in ax:
        common.square(a_)
    fig.tight_layout(w_pad=2.4)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
