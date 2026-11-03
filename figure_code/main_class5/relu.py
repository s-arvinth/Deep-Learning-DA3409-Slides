"""relu — The ReLU activation h(a) = max(0, a)

Used in: Class 5 (B), slide(s): "The activation function: ReLU"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python relu.py            -> output/relu.pdf
Ported from: Class05_Shallow_NN_Regression/B_Recreated_Figures/src/figures/fig05_relu.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import OUTC, GREY, relu

NAME = "relu"


def build():
    common.use()
    a = np.linspace(-2, 2, 600)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot(a, a, color=GREY, lw=1.3, ls="--", label=r"$a$  (identity)")
    ax.plot(a, relu(a), color=OUTC, lw=2.8, label=r"$h(a)=\max(0,a)$")
    ax.axhline(0, color=GREY, lw=0.6, ls=":")
    ax.axvline(0, color=GREY, lw=0.6, ls=":")
    ax.set_ylim(-1.05, 2.1)
    ax.annotate("inactive", xy=(-1.5, 0.12), color=GREY, fontsize=14.3)
    ax.annotate("active", xy=(0.75, 0.45), color=OUTC, fontsize=14.3)
    ax.legend(loc="upper left", fontsize=14.3)
    ax.set(xlabel=r"pre-activation  $a$", ylabel=r"activation  $h(a)$",
           title=r"the rectified linear unit")
    ax.title.set_fontsize(12.5)
    common.square(ax)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
