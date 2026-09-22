"""The bidirectional RNN, drawn twice and measured once.

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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from common import style as S
from common import draw as D

NAME = "bidirectional_goodfellow"
NAMES = ("bidirectional_goodfellow", "bidirectional_jurafsky", "influence_maps")


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
    S.save(fig, "bidirectional_goodfellow")


def build_jurafsky():
    S.use()
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    xs = [0, 1.7, 3.4, 6.0]; labs = ["1", "2", "3", "n"]
    # the two bands
    for y, col, fill, lab in ((0.0, D.ENC, D.ENC_F, "RNN 1  (left to right)"), (1.5, S.L2, "#D6E6EC", "RNN 2  (right to left)")):
        ax.add_patch(FancyBboxPatch((-0.6, y - 0.42), 7.2, 0.84, boxstyle="round,pad=0.05,rounding_size=0.3", fc=fill, ec=col, lw=1.4, alpha=0.85, zorder=1))
        ax.text(7.0, y, lab, ha="left", va="center", fontsize=12, color=col)
    for k, (x, lab) in enumerate(zip(xs, labs)):
        for y, col in ((0.0, D.ENC), (1.5, S.L2)):
            ax.add_patch(Rectangle((x - 0.2, y - 0.24), 0.4, 0.48, fc="white", ec=col, lw=1.3, zorder=3))
        # inputs, up into both
        ax.add_patch(Rectangle((x - 0.32, -1.45), 0.64, 0.5, fc="#F2F2F2", ec=S.GREY, lw=1.0, zorder=3)); ax.text(x, -1.2, "$\\mathbf{x}_{%s}$" % lab, ha="center", va="center", fontsize=11)
        D.arrow(ax, (x - 0.08, -0.93), (x - 0.08, -0.26), lw=1.1)
        ax.annotate("", xy=(x + 0.45, 1.26), xytext=(x + 0.12, -0.93), arrowprops=dict(arrowstyle="-|>", color=S.L2, lw=1.0, connectionstyle="arc3,rad=-0.3", shrinkA=0, shrinkB=0), zorder=2)
        # outputs, concatenated
        ax.add_patch(Rectangle((x - 0.3, 2.55), 0.6, 0.4, fc=D.DEC_F, ec=D.DEC, lw=1.1, zorder=3))
        D.arrow(ax, (x - 0.08, 1.76), (x - 0.08, 2.52), color=S.L2, lw=1.1)
        ax.annotate("", xy=(x - 0.22, 2.52), xytext=(x - 0.45, 0.26), arrowprops=dict(arrowstyle="-|>", color=D.ENC, lw=1.0, connectionstyle="arc3,rad=0.25", shrinkA=0, shrinkB=0), zorder=2)
        D.arrow(ax, (x, 2.97), (x, 3.45), color=D.DEC, lw=1.2)
        ax.text(x, 3.7, "$\\mathbf{y}_{%s}$" % lab, ha="center", va="center", fontsize=11.5, color=D.DEC)
        if k < 3:
            nxt = xs[k + 1]
            D.arrow(ax, (x + 0.22, 0.0), (nxt - 0.22, 0.0), color=D.ENC, lw=1.4)
            D.arrow(ax, (nxt - 0.22, 1.5), (x + 0.22, 1.5), color=S.L2, lw=1.4)
    ax.text(4.7, 0.0, "$\\cdots$", ha="center", va="center", fontsize=14, color=D.ENC, zorder=4)
    ax.text(4.7, 1.5, "$\\cdots$", ha="center", va="center", fontsize=14, color=S.L2, zorder=4)
    ax.text(4.7, -1.2, "$\\cdots$", ha="center", va="center", fontsize=14)
    ax.text(7.0, 2.75, "concatenated outputs $[\\mathbf{h}^f_t ; \\mathbf{h}^b_t]$", ha="left", va="center", fontsize=11.5, color=D.DEC)
    ax.set_xlim(-1.0, 10.8); ax.set_ylim(-1.8, 4.1); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "bidirectional_jurafsky")


def build_maps():
    from common import experiment as E
    S.use()
    out = E.run()
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
    S.save(fig, "influence_maps")


def build():
    build_goodfellow(); build_jurafsky(); build_maps()


if __name__ == "__main__":
    build()
