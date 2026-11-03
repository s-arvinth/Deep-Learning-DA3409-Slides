"""The four normalizers differ only in which entries share a mean.

Each panel is the same block of activations: rows are the members of a
mini-batch, columns are the hidden units. The shaded cells are the ones
that are pooled to compute a single mean and variance.

Batch normalization pools down a column, layer normalization along a
row, group normalization along part of a row, and instance
normalization over a single cell -- which for a fully connected layer
does nothing at all.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "norm_axes"

B, M = 6, 8


def build():
    S.use()
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.4))
    cases = [
        ("batch norm", lambda r, c: c == 3, S.OUTC,
         "one mean per unit,\nover the batch"),
        ("layer norm", lambda r, c: r == 2, S.L2,
         "one mean per example,\nover the units"),
        ("group norm", lambda r, c: r == 2 and 2 <= c < 6, S.ACC,
         "one mean per example\nand group"),
        ("instance norm", lambda r, c: r == 2 and c == 3, S.L1,
         "one mean per example\nand unit"),
    ]
    for ax, (title, sel, col, note) in zip(axes, cases):
        for r in range(B):
            for c in range(M):
                on = sel(r, c)
                ax.add_patch(plt.Rectangle(
                    (c, -r), 0.88, 0.88,
                    facecolor=col if on else "#EDEDED",
                    alpha=0.95 if on else 1.0,
                    edgecolor="white", lw=1.6))
        ax.set_xlim(-0.4, M + 0.2)
        ax.set_ylim(-B + 0.1, 2.2)
        ax.axis("off")
        ax.annotate("hidden units", xy=(M / 2, 1.15), fontsize=14,
                    ha="center", color=S.GREY)
        ax.annotate("", xy=(M - 0.1, 0.95), xytext=(0.1, 0.95),
                    arrowprops=dict(arrowstyle="->", color=S.GREY, lw=1.2))
        ax.annotate("mini-batch", xy=(-0.62, -B / 2 + 0.5), fontsize=14,
                    rotation=90, va="center", ha="center", color=S.GREY)
        ax.set_title("%s\n%s" % (title, note), fontsize=15)
    fig.tight_layout(w_pad=1.2)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
