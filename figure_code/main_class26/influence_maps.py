"""influence_maps — ||d o_t / d e_s|| maps for a one- and a two-directional RNN

Used in: Class 26 (B_Recreated_Figures), slide(s): "Both directions, measured"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python influence_maps.py            -> output/influence_maps.pdf
         python influence_maps.py --out DIR  -> DIR/influence_maps.pdf

Ported from Class26_BPTT_Bidirectional_RNNs/B_Recreated_Figures/src/figures/fig08_bidirectional.py (split: that module also wrote bidirectional_goodfellow, bidirectional_jurafsky; this script writes only influence_maps).
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

NAME = "influence_maps"


def node(ax, x, y, label, color, fill, r=0.36, fs=11.5):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.3, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def build_maps():
    import experiment as E
    S.use()
    out = E.run(stages=("influence",))    # only the stage this figure needs
    fig, (b, c) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, key, title in ((b, "uni", "(a) one direction: causal"), (c, "bi", "(b) two directions: the whole input")):
        M = out["influence"][key]; T = M.shape[0]
        im = ax.imshow(M / M.max(), cmap="Purples", vmin=0, vmax=1, origin="upper")
        ax.set_xlabel("input step $s$"); ax.set_ylabel("output step $t$")
        ax.set_xticks(range(T)); ax.set_xticklabels([str(i + 1) for i in range(T)])
        ax.set_yticks(range(T)); ax.set_yticklabels([str(i + 1) for i in range(T)])
        ax.set_title(title, loc="left")
    b.text(5.6, 1.2, "zero: $\\mathbf{x}_s$ with $s > t$\ncannot reach $\\mathbf{o}_t$", ha="center", va="center", fontsize=10.5, color=S.L1)
    cb = fig.colorbar(im, ax=c, fraction=0.046, pad=0.04); cb.set_label("$\\|\\partial \\mathbf{o}_t / \\partial \\mathbf{e}_s\\|$, relative to the largest")
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


def build():
    build_maps()


if __name__ == "__main__":
    build()
