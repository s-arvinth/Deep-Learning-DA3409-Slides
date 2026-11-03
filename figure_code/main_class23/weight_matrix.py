"""weight_matrix — A convolution is a fully connected layer with most of it deleted

Used in: Class 23 (B_Recreated_Figures), slide(s): "A convolution is a dense layer with most of it deleted"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python weight_matrix.py            -> output/weight_matrix.pdf
         python weight_matrix.py --out DIR  -> DIR/weight_matrix.pdf

Ported from Class23_Convolution_Filters/B_Recreated_Figures/src/figures/fig04_weight_matrix.py.
Original description:

A convolution is a fully connected layer with most of it deleted.

Following the construction of Prince (2023), Fig. 10.4. Each panel is
the actual matrix that the layer applies, built by `conv_matrix` and
checked against the implementation of the convolution itself.

(a) A dense layer on twelve inputs: every entry is free, 144 of them.
(b) The same layer restricted to a kernel of size three: the matrix is
    banded, and the three colours repeat down the band because the
    weights are TIED, not merely sparse. Three free numbers.
(c) Adding a stride of two removes every other row, halving the output
    length and leaving the same three free numbers.

Sparsity alone would give a layer with 3n free weights. Sharing is what
brings it down to 3.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import common as S
from common import conv_matrix

NAME = "weight_matrix"

N = 12
W = np.array([0.9, -0.5, 0.35])


def build():
    S.use()
    rng = np.random.default_rng(1)
    dense = rng.normal(size=(N, N)) * 0.5

    mats = [(dense, "(a) dense", N * N, N * N),
            (conv_matrix(N, W, s=1, p=1), r"(b) kernel $3$, stride $1$",
             int(np.count_nonzero(conv_matrix(N, W, s=1, p=1))), 3),
            (conv_matrix(N, W, s=2, p=1), r"(c) kernel $3$, stride $2$",
             int(np.count_nonzero(conv_matrix(N, W, s=2, p=1))), 3)]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    vmax = 1.0
    for ax, (M, title, nz, free) in zip(axes, mats):
        ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("input position $j$")
        ax.set_ylabel("output position $i$")
        ax.set_title(title)
        ax.annotate("non-zero entries: %d\nfree parameters: %d" % (nz, free),
                    xy=(0.5, -0.13), xycoords="axes fraction", ha="center",
                    va="top", fontsize=13, color=S.OUTC)
    fig.tight_layout(w_pad=1.6)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
