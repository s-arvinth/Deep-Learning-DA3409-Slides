"""Training an RNN language model, in the layout of Jurafsky & Martin
(2026), Fig. 14.6: input embeddings, the RNN band, a softmax over the
vocabulary at every step, the per-word loss -log y_hat[next word], and
the next word above; the total loss is the average over the sequence."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import style as S
from common import bookdraw as B

NAME = "rnn_lm_training"
WORDS = ["So", "long", "and", "thanks", "for"]
NEXT = ["long", "and", "thanks", "for", "all"]


def build():
    S.use()
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    xs = [k * 1.7 for k in range(5)]
    yw, ye, yh, ys, yl, yn = 0.0, 1.15, 2.5, 3.75, 4.75, 5.55
    B.band(ax, -0.8, xs[-1] + 1.6, yh, h=1.05, color=B.SRC, label="RNN", lx=-0.65)
    for k, (x, w, nx) in enumerate(zip(xs, WORDS, NEXT)):
        B.column(ax, x, yw, ye, yh, ys, word=w, color="black", rng=rng)
        ax.text(x - 0.42, yh + 0.55, "$\\mathbf{V}\\mathbf{h}$", ha="right", va="center", fontsize=9, color=S.GREY) if k == 0 else None
        # loss box
        ax.add_patch(Rectangle((x - 0.62, yl - 0.22), 1.24, 0.44, fc=B.TGT_F, ec=B.TGT, lw=1.1, zorder=3))
        ax.text(x, yl, "$-\\log \\hat{y}_{\\mathrm{%s}}$" % nx, ha="center", va="center", fontsize=9, color=B.TGT, zorder=4)
        B.arrow(ax, (x, ys + 0.2), (x, yl - 0.24), color="black", lw=0.9)
        ax.text(x, yn, nx, ha="center", va="center", fontsize=11.5, color=B.TGT)
        B.arrow(ax, (x, yn - 0.22), (x, yl + 0.24), color=B.TGT, lw=0.9)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.SRC, lw=1.3)
            ax.text((xs[k - 1] + x) / 2, yh + 0.2, "$\\mathbf{h}$", ha="center", fontsize=9, color=B.SRC) if k == 1 else None
    B.arrow(ax, (xs[-1] + 0.27, yh), (xs[-1] + 1.4, yh), color=B.SRC, lw=1.3)
    ax.text(xs[-1] + 1.15, yh, "", fontsize=1)
    for y in (ye, yl, yn):
        ax.text(xs[-1] + 1.15, y, "$\\cdots$", ha="center", va="center", fontsize=13, color=S.GREY)
    ax.text(xs[-1] + 2.3, yl, "$\\dfrac{1}{T}\\sum_{t=1}^{T} L_{\\mathrm{CE}}$", ha="left", va="center", fontsize=13, color=B.TGT)
    ax.text(xs[-1] + 2.3, yl - 0.75, "the loss of the sequence:\nthe average per-word loss", ha="left", va="center", fontsize=9.5, color=S.GREY)
    for y, lab in ((ye, "input\nembeddings $\\mathbf{e}$"), (ys, "softmax over\nvocabulary $\\hat{\\mathbf{y}}$"), (yl, "loss"), (yn, "next word")):
        ax.text(-1.05, y, lab, ha="right", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-3.6, xs[-1] + 5.2); ax.set_ylim(-0.6, 6.1); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
