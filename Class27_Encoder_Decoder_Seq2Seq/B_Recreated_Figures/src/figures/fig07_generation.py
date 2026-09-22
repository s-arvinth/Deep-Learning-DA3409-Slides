"""Autoregressive generation, in the layout of Jurafsky & Martin (2026),
Fig. 14.9: input word, embedding, the RNN band, softmax, sampled word;
the sampled word at each step becomes the input word at the next."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import style as S
from common import bookdraw as B

NAME = "generation"
IN = ["$\\langle s\\rangle$", "So", "long", "and"]
OUT = ["So", "long", "and", "?"]


def build():
    S.use()
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    xs = [0, 1.9, 3.8, 5.7]
    yw, ye, yh, ys, yo = 0.0, 1.1, 2.4, 3.6, 4.6
    B.band(ax, -1.9, 7.4, yh, h=1.0, color=B.SRC, label="RNN", lx=-1.7)
    for k, (x, wi, wo) in enumerate(zip(xs, IN, OUT)):
        B.column(ax, x, yw, ye, yh, ys, yo, word=wi, out=wo, color="black", rng=rng, out_color=B.TGT)
        if k > 0:
            B.arrow(ax, (xs[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.SRC, lw=1.4)
            # the sampled word becomes the next input: one slanted dashed line, as in the book
            B.arrow(ax, (xs[k - 1] + 0.45, yo - 0.15), (x - 0.45, yw + 0.15), color=B.TGT, lw=1.1, ls=(0, (4, 2)), z=2)
    B.arrow(ax, (xs[-1] + 0.27, yh), (7.2, yh), color=B.SRC, lw=1.4)
    for y, lab in ((yw, "input word"), (ye, "embedding"), (ys, "softmax"), (yo, "sampled word")):
        ax.text(-2.05, y, lab, ha="right", va="center", fontsize=11, color=S.GREY)
    ax.text(3.0, 5.35, "each sampled word is fed back as the next input (dashed); stop at $\\langle/s\\rangle$ or a length limit", ha="center", fontsize=11, color=B.TGT)
    ax.set_xlim(-4.2, 7.8); ax.set_ylim(-0.7, 5.8); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
