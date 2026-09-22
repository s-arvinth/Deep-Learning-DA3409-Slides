"""The recurrent network reading the running sentence, one word at a
time: at each step the cell takes the current word and the state, emits
a prediction, and hands the new state on.  Schematic: no numbers."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from common import style as S
from common import draw as D

NAME = "running_example"
WORDS = ["The", "flights", "the", "airline", "was", "canceling"]
NEXT = ["flights", "the", "airline", "was", "canceling", "were"]


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    xs = [k * 1.7 for k in range(len(WORDS))]
    ax.add_patch(FancyBboxPatch((-1.55, 1.05), 1.1, 0.8, boxstyle="round,pad=0.05,rounding_size=0.15", fc="white", ec=S.GREY, lw=1.1, ls=(0, (3, 2)), zorder=3))
    ax.text(-1.0, 1.45, "$\\mathbf{h}_0 = \\mathbf{0}$", ha="center", va="center", fontsize=10, color=S.GREY)
    prev = -1.0
    for k, (x, w, nx) in enumerate(zip(xs, WORDS, NEXT)):
        t = k + 1
        ax.add_patch(FancyBboxPatch((x - 0.6, 1.05), 1.2, 0.8, boxstyle="round,pad=0.05,rounding_size=0.15", fc=D.ENC_F, ec=D.ENC, lw=1.4, zorder=3))
        ax.text(x, 1.45, "$\\mathbf{h}_{%d}$" % t, ha="center", va="center", fontsize=12, color=D.ENC)
        ax.add_patch(Rectangle((x - 0.6, -0.35), 1.2, 0.55, fc="#F2F2F2", ec=S.GREY, lw=1.0, zorder=3))
        ax.text(x, -0.07, w, ha="center", va="center", fontsize=11)
        ax.text(x, -0.75, "$\\mathbf{x}_{%d}$" % t, ha="center", va="center", fontsize=10, color=S.GREY)
        D.arrow(ax, (x, 0.24), (x, 1.0))
        ax.add_patch(Rectangle((x - 0.6, 2.55), 1.2, 0.55, fc=D.DEC_F, ec=D.DEC, lw=1.1, zorder=3))
        ax.text(x, 2.82, nx, ha="center", va="center", fontsize=10.5, color=D.DEC)
        ax.text(x, 3.4, "$\\hat{\\mathbf{y}}_{%d}$: next word?" % t, ha="center", va="center", fontsize=9.5, color=D.DEC)
        D.arrow(ax, (x, 1.9), (x, 2.5), color=D.DEC)
        D.arrow(ax, (prev + 0.65 if k else prev + 0.6, 1.45), (x - 0.65, 1.45), color=S.OUTC, lw=1.8)
        prev = x
    ax.text(xs[2] + 0.4, -1.45, "“flights” enters the state at $\\mathbf{h}_2$ and is carried along the maroon wire;",
            ha="center", fontsize=11, color=S.OUTC)
    ax.text(xs[2] + 0.4, -1.85, "it can still be in $\\mathbf{h}_6$ when the next word, “were”, must be predicted", ha="center", fontsize=11, color=S.OUTC)
    ax.text(xs[-1] + 1.05, 1.45, "$\\mathbf{h}_6$", ha="left", va="center", fontsize=12, color=S.OUTC)
    ax.set_xlim(-1.9, xs[-1] + 1.8); ax.set_ylim(-2.2, 3.9); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
