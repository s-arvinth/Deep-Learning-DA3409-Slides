"""sigmoid — The logistic sigmoid and its derivative

Used in: Class 6 (B), slide(s): "Step 2: the logistic sigmoid"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python sigmoid.py            -> output/sigmoid.pdf
Ported from: Class06_Shallow_NN_Classification/B_Recreated_Figures/src/figures/fig06_sigmoid.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import L2, OUTC, ACC, GREY, sigmoid, dsigmoid

NAME = "sigmoid"


def build():
    common.use()
    a = np.linspace(-8, 8, 900)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.5))

    ax[0].plot(a, sigmoid(a), color=OUTC, lw=2.8,
               label=r"$\sigma(a)=1/(1+e^{-a})$")
    ax[0].axhline(0.5, color=GREY, lw=0.8, ls=":")
    ax[0].axvline(0.0, color=GREY, lw=0.8, ls=":")
    ax[0].axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax[0].annotate(r"$\sigma(0)=\frac{1}{2}$", xy=(0.4, 0.53), fontsize=15.6,
                   color=GREY)
    ax[0].legend(loc="upper left", fontsize=14.3)
    ax[0].set(xlabel=r"$a$", ylabel=r"$\hat y=\sigma(a)$",
              ylim=(-0.06, 1.12))
    ax[0].set_title(r"(a)  maps $\mathbb{R}$ onto $(0,1)$", fontsize=16.2)

    ax[1].plot(a, dsigmoid(a), color=L2, lw=2.8,
               label=r"$\sigma'(a)=\sigma(a)(1-\sigma(a))$")
    ax[1].fill_between(a, 0, dsigmoid(a), where=np.abs(a) > 4,
                       color=ACC, alpha=.30, lw=0)
    ax[1].annotate("saturated:\n" r"$\sigma'\approx 0$", xy=(-7.6, 0.055),
                   fontsize=14.3, color=ACC)
    ax[1].annotate("saturated", xy=(4.7, 0.045), fontsize=14.3, color=ACC)
    ax[1].legend(loc="upper left", fontsize=13.7)
    ax[1].set(xlabel=r"$a$", ylabel=r"$\sigma'(a)$", ylim=(0, 0.32))
    ax[1].set_title(r"(b)  the gradient dies in the tails", fontsize=16.2)
    for a_ in ax:
        common.square(a_)
    fig.tight_layout(w_pad=2.4)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
