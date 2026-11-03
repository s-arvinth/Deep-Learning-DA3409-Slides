"""bernoulli — The Bernoulli distribution: one parameter, the probability of y = 1

Used in: Class 6 (B), slide(s): "Step 1: the Bernoulli distribution"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python bernoulli.py            -> output/bernoulli.pdf
Ported from: Class06_Shallow_NN_Classification/B_Recreated_Figures/src/figures/fig05_bernoulli.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import L1, L2, OUTC

NAME = "bernoulli"


def build():
    common.use()
    vals = [0.20, 0.50, 0.80]
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.9))
    for a_, p, c in zip(ax, vals, [L1, L2, OUTC]):
        a_.bar([0, 1], [1 - p, p], width=0.42, color=c, alpha=.85,
               edgecolor="white", lw=1.2)
        for xk, h in zip([0, 1], [1 - p, p]):
            a_.text(xk, h + 0.035, f"{h:.2f}", ha="center", fontsize=15.0,
                    color=c)
        a_.set(xticks=[0, 1], ylim=(0, 1.12), xlabel=r"$y$")
        a_.set_title(rf"$\hat y={p:.1f}$", fontsize=16.9)
        common.square(a_)
    ax[0].set_ylabel(r"$p(y\mid\hat y)$")
    fig.suptitle(r"$p(y\mid\hat y)=\hat y^{\,y}(1-\hat y)^{1-y}$,"
                 r"   $y\in\{0,1\}$", fontsize=17.6, y=1.04)
    fig.tight_layout(w_pad=2.2)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
