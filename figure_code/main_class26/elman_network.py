"""elman_network — the Elman network in the layout of Jurafsky & Martin Fig. 14.1

Used in: Class 26 (B_Recreated_Figures), slide(s): "The simple recurrent network, drawn"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python elman_network.py            -> output/elman_network.pdf
         python elman_network.py --out DIR  -> DIR/elman_network.pdf

Ported from Class26_BPTT_Bidirectional_RNNs/B_Recreated_Figures/src/figures/fig00_elman.py (split: that module also wrote elman_feedforward; this script writes only elman_network).
Original description:

The Elman network, in the layouts of Jurafsky & Martin (2026), Figs.
14.1 and 14.2: (a) the three layers with the recurrent link from the
hidden layer back to its own input, dashed; (b) the same network drawn
as a feedforward step, the previous hidden layer multiplied by U and
added to the feedforward component W x_t.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import common as S
import common as D

NAME = "elman_network"


def layer(ax, x, y, label, color, fill, w=0.7, h=2.4, n=4, vertical=True):
    if vertical:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.3", fc=fill, ec=color, lw=1.4, zorder=3))
        for k in range(n):
            yy = y - h / 2 + (k + 0.5) * h / n
            ax.add_patch(plt.Circle((x, yy), 0.16, fc="white", ec=color, lw=1.0, zorder=4))
        ax.text(x, y - h / 2 - 0.45, label, ha="center", va="center", fontsize=13, color=color)
    else:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fill, ec=color, lw=1.4, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=12, color=color, zorder=5)


def build_a():
    S.use()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    layer(ax, 0, 0, "$\\mathbf{x}_t$", S.GREY, "#EEEEEE", h=2.8, n=5)
    layer(ax, 3.0, 0, "$\\mathbf{h}_t$", D.ENC, D.ENC_F, h=2.0, n=3)
    layer(ax, 6.0, 0, "$\\mathbf{y}_t$", D.DEC, D.DEC_F, h=1.6, n=2)
    D.arrow(ax, (0.4, 0), (2.6, 0), color="black", lw=1.4); ax.text(1.5, 0.3, "$\\mathbf{W}$", ha="center", fontsize=12)
    D.arrow(ax, (3.4, 0), (5.6, 0), color="black", lw=1.4); ax.text(4.5, 0.3, "$\\mathbf{V}$", ha="center", fontsize=12)
    # the recurrent link: out of the top of h, round, back into its input side
    ax.plot([3.0, 3.0, 1.9, 1.9], [1.05, 1.9, 1.9, 0.55], color=D.ENC, lw=1.5, ls=(0, (4, 2)), zorder=2)
    ax.annotate("", xy=(2.62, 0.35), xytext=(1.9, 0.55), arrowprops=dict(arrowstyle="-|>", color=D.ENC, lw=1.5, ls=(0, (4, 2)), shrinkA=0, shrinkB=0), zorder=2)
    ax.text(2.45, 2.2, "$\\mathbf{U}$: the hidden layer's own value\nfrom the previous step, $\\mathbf{h}_{t-1}$", ha="center", va="bottom", fontsize=11, color=D.ENC)
    ax.set_xlim(-1.0, 7.2); ax.set_ylim(-2.3, 3.6); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


def build():
    build_a()


if __name__ == "__main__":
    build()
