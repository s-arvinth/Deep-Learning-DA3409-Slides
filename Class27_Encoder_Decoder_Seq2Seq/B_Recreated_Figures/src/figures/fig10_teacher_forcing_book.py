"""Training with teacher forcing, in the layout of Jurafsky & Martin
(2026), Fig. 14.19: the encoder reads the source, the decoder reads the
gold target tokens whatever it predicted, and a cross-entropy loss is
paid at every target position against the gold answer; the total loss
is their average."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import style as S
from common import bookdraw as B

NAME = "teacher_forcing_book"
SRCW = ["the", "green", "witch", "arrived"]
TGTW = ["llegó", "la", "bruja", "verde"]


def build():
    S.use()
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    dx = 1.45
    xs = [k * dx for k in range(4)]; xt = [xs[-1] + 1.4 * dx + k * dx for k in range(5)]
    yw, ye, yh, ys, yl, yg = 0.0, 1.05, 2.3, 3.5, 4.55, 5.6
    B.band(ax, xs[0] - 0.6, xs[-1] + 0.6, yh, h=1.0, color=B.SRC)
    B.band(ax, xt[0] - 0.6, xt[-1] + 0.6, yh, h=1.0, color=B.TGT)
    for k, (x, w) in enumerate(zip(xs, SRCW)):
        B.column(ax, x, yw, ye, yh, word=w, color=B.SRC)
        ax.text(x, yw - 0.42, "$\\mathbf{x}_%d$" % (k + 1), ha="center", fontsize=9, color=S.GREY)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.SRC, lw=1.3)
    B.arrow(ax, (xs[-1] + 0.27, yh), (xt[0] - 0.27, yh), color=B.SRC, lw=1.3, ls=(0, (4, 2)))
    tin = ["$\\langle s\\rangle$"] + TGTW
    gold = TGTW + ["$\\langle/s\\rangle$"]
    for k, (x, wi, wg) in enumerate(zip(xt, tin, gold)):
        B.column(ax, x, yw, ye, yh, ys, word=wi, color=B.TGT, rng=rng)
        if k:
            B.arrow(ax, (xt[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.TGT, lw=1.3)
        # per-word loss box
        ax.add_patch(Rectangle((x - 0.55, yl - 0.3), 1.1, 0.6, fc=B.TGT_F, ec=B.TGT, lw=1.2, zorder=3))
        ax.text(x, yl + 0.1, "$L_%d =$" % (k + 1), ha="center", va="center", fontsize=8.5, color=B.TGT, zorder=4)
        ax.text(x, yl - 0.13, "$-\\log P(y_%d)$" % (k + 1), ha="center", va="center", fontsize=8.5, color=B.TGT, zorder=4)
        B.arrow(ax, (x, ys + 0.2), (x, yl - 0.32), color="black", lw=0.9)
        ax.text(x, yg, wg, ha="center", va="center", fontsize=11.5, color=B.TGT)
        ax.text(x, yg - 0.38, "$y_%d$" % (k + 1), ha="center", fontsize=9, color=S.GREY)
        B.arrow(ax, (x, yg - 0.55), (x, yl + 0.32), color=B.TGT, lw=0.9)
    B.brace(ax, xs[0] - 0.4, xs[-1] + 0.4, yw - 0.75, "encoder", color=B.SRC)
    B.brace(ax, xt[0] - 0.4, xt[-1] + 0.4, yg + 0.35, "decoder", color=B.TGT, up=True)
    for y, lab in ((ye, "embedding\nlayer"), (yh, "hidden\nlayer(s)"), (ys, "softmax $\\hat{\\mathbf{y}}$"), (yl, "per-word\nloss"), (yg, "gold\nanswers")):
        ax.text(xt[-1] + 0.85, y, lab, ha="left", va="center", fontsize=10.5, color=S.GREY)
    ax.text(xs[1] + 0.3, 4.9, "total loss is the average\ncross-entropy loss per\ntarget word:", ha="center", va="center", fontsize=10.5, color=S.GREY)
    ax.text(xs[1] + 0.3, 3.75, "$L = \\dfrac{1}{T}\\sum_{i=1}^{T} L_i$", ha="center", va="center", fontsize=14, color=B.TGT)
    ax.set_xlim(xs[0] - 1.0, xt[-1] + 3.0); ax.set_ylim(-1.7, 6.7); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
