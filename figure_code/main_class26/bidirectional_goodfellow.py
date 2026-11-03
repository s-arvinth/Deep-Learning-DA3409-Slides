"""bidirectional_goodfellow — the bidirectional RNN in the layout of Goodfellow et al. Fig. 10.11

Used in: Class 26 (B_Recreated_Figures), slide(s): "Both directions, drawn"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python bidirectional_goodfellow.py            -> output/bidirectional_goodfellow.pdf
         python bidirectional_goodfellow.py --out DIR  -> DIR/bidirectional_goodfellow.pdf

Ported from Class26_BPTT_Bidirectional_RNNs/B_Recreated_Figures/src/figures/fig08_bidirectional.py (split: that module also wrote bidirectional_jurafsky, influence_maps; this script writes only bidirectional_goodfellow).
Original description:

The bidirectional RNN, drawn twice and measured once.

`bidirectional_goodfellow`  the layout of Goodfellow et al. (2016), Fig.
    10.11: a forward chain h, a backward chain g, both feeding o_t, with
    Jurafsky's names h^f, h^b for the two states.
`bidirectional_jurafsky`  the layout of Jurafsky & Martin (2026), Fig.
    14.11: RNN 1 left to right and RNN 2 right to left, their outputs
    concatenated at each step.
`influence_maps`  ||d o_t / d e_s|| for every input step s and output step
    t, a one-directional and a two-directional network at initialisation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import common as S
import common as D

NAME = "bidirectional_goodfellow"


def node(ax, x, y, label, color, fill, r=0.36, fs=11.5):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.3, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def build_goodfellow():
    S.use()
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    xs = [0, 2.2, 4.4]; st = ["t-1", "t", "t+1"]
    for x, s in zip(xs, st):
        node(ax, x, 0, "$\\mathbf{x}_{%s}$" % s, S.GREY, "#EEEEEE")
        node(ax, x, 1.5, "$\\mathbf{h}^f_{%s}$" % s, D.ENC, D.ENC_F)
        node(ax, x, 3.0, "$\\mathbf{h}^b_{%s}$" % s, S.L2, "#D6E6EC")
        node(ax, x, 4.5, "$\\mathbf{o}_{%s}$" % s, D.DEC, D.DEC_F)
        node(ax, x, 5.7, "$L_{%s}$" % s, S.ACC, "#F5EBD0", r=0.3)
        node(ax, x, 6.8, "$\\mathbf{y}_{%s}$" % s, S.GREY, "white", r=0.3)
        # x to both chains: straight up to h^f, and a curved path round to h^b
        D.arrow(ax, (x, 0.38), (x, 1.12), lw=1.2)
        ax.annotate("", xy=(x + 0.3, 2.78), xytext=(x + 0.3, 0.22),
                    arrowprops=dict(arrowstyle="-|>", color=S.L2, lw=1.0, connectionstyle="arc3,rad=-0.45", shrinkA=0, shrinkB=0), zorder=2)
        # both chains to o: h^f curved on the left, h^b straight
        ax.annotate("", xy=(x - 0.3, 4.2), xytext=(x - 0.3, 1.78),
                    arrowprops=dict(arrowstyle="-|>", color=D.ENC, lw=1.0, connectionstyle="arc3,rad=0.45", shrinkA=0, shrinkB=0), zorder=2)
        D.arrow(ax, (x, 3.38), (x, 4.12), color=S.L2, lw=1.2)
        D.arrow(ax, (x, 4.88), (x, 5.4), lw=1.2); D.arrow(ax, (x, 6.5), (x, 6.0), lw=1.2)
    for p, q in zip(xs[:-1], xs[1:]):
        D.arrow(ax, (p + 0.38, 1.5), (q - 0.38, 1.5), color=D.ENC, lw=1.5)
        D.arrow(ax, (q - 0.38, 3.0), (p + 0.38, 3.0), color=S.L2, lw=1.5)
    D.arrow(ax, (xs[-1] + 0.38, 1.5), (xs[-1] + 1.1, 1.5), color=D.ENC, lw=1.5); D.arrow(ax, (xs[0] - 1.1, 1.5), (xs[0] - 0.38, 1.5), color=D.ENC, lw=1.5)
    D.arrow(ax, (xs[0] - 0.38, 3.0), (xs[0] - 1.1, 3.0), color=S.L2, lw=1.5); D.arrow(ax, (xs[-1] + 1.1, 3.0), (xs[-1] + 0.38, 3.0), color=S.L2, lw=1.5)
    ax.text(5.7, 1.5, "forward chain: reads $\\mathbf{x}_1 \\ldots \\mathbf{x}_t$", fontsize=11.5, color=D.ENC, va="center")
    ax.text(5.7, 3.0, "backward chain: reads $\\mathbf{x}_n \\ldots \\mathbf{x}_t$", fontsize=11.5, color=S.L2, va="center")
    ax.text(5.7, 4.5, "output from both states", fontsize=11.5, color=D.DEC, va="center")
    ax.set_xlim(-1.5, 10.5); ax.set_ylim(-0.6, 7.4); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


def build():
    build_goodfellow()


if __name__ == "__main__":
    build()
