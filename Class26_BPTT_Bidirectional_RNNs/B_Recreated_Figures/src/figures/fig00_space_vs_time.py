"""From space to time: a convolution slides one kernel over the positions
of a grid and computes each output from a local window; a recurrent
network slides one cell over the steps of a sequence and, in addition,
passes a state from each step to the next.  Same idea -- one set of
weights reused everywhere -- with one new wire."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from common import style as S
from common import draw as D

NAME = "space_vs_time"


def build():
    S.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # (a) convolution over space
    n = 7
    for i in range(n):
        for j in range(n):
            a.add_patch(Rectangle((i, j), 1, 1, fc="#F4F4F4", ec="#BBBBBB", lw=0.6))
    for i in range(2, 5):
        for j in range(3, 6):
            a.add_patch(Rectangle((i, j), 1, 1, fc=D.ENC_F, ec=D.ENC, lw=1.2, zorder=3))
    a.add_patch(Rectangle((2, 3), 3, 3, fc="none", ec=D.ENC, lw=2.0, zorder=4))
    a.text(3.5, 6.4, "one kernel, $3 \\times 3$", ha="center", fontsize=11.5, color=D.ENC)
    for (i, j) in ((0.5, 0.5), (5.5, 1.5), (0.5, 5.5)):
        a.add_patch(Rectangle((i - 0.5, j - 0.5), 3, 3, fc="none", ec=D.ENC, lw=1.0, ls=(0, (3, 2)), zorder=4, alpha=0.6))
    a.annotate("", xy=(6.0, 4.5), xytext=(4.8, 4.5), arrowprops=dict(arrowstyle="-|>", color=D.ENC, lw=1.4))
    # output pixel
    a.add_patch(Rectangle((8.2, 4.0), 1, 1, fc=D.DEC_F, ec=D.DEC, lw=1.2, zorder=3))
    a.text(8.7, 5.4, "output at\none position", ha="center", fontsize=10.5, color=D.DEC)
    a.annotate("", xy=(8.2, 4.5), xytext=(6.2, 4.5), arrowprops=dict(arrowstyle="-|>", color=D.DEC, lw=1.2))
    a.text(4.5, -0.9, "the same weights at every position; each output\nsees only its window", ha="center", fontsize=11, color=S.GREY)
    a.set_xlim(-0.5, 9.8); a.set_ylim(-1.7, 7.2); a.set_aspect("equal"); a.axis("off")
    a.set_title("(a) convolution: sharing across space", loc="left")
    # (b) recurrence over time
    xs = [0, 1.6, 3.2, 4.8, 6.4]
    words = ["The", "flights", "the", "airline", "$\\cdots$"]
    prev = None
    for x, w in zip(xs, words):
        b.add_patch(FancyBboxPatch((x - 0.55, 1.2), 1.1, 0.8, boxstyle="round,pad=0.05,rounding_size=0.15", fc=D.ENC_F, ec=D.ENC, lw=1.4, zorder=3))
        b.text(x, 1.6, "cell", ha="center", va="center", fontsize=11, color=D.ENC)
        b.text(x, 0.35, w, ha="center", va="center", fontsize=11.5)
        D.arrow(b, (x, 0.65), (x, 1.15))
        b.add_patch(Rectangle((x - 0.4, 2.7), 0.8, 0.55, fc=D.DEC_F, ec=D.DEC, lw=1.1, zorder=3))
        b.text(x, 2.97, "$\\hat{\\mathbf{y}}$", ha="center", va="center", fontsize=10.5, color=D.DEC)
        D.arrow(b, (x, 2.05), (x, 2.65), color=D.DEC)
        if prev is not None:
            D.arrow(b, (prev + 0.6, 1.6), (x - 0.6, 1.6), color=S.OUTC, lw=1.8)
            b.text((prev + x) / 2, 1.9, "$\\mathbf{h}$", ha="center", fontsize=11, color=S.OUTC)
        prev = x
    b.text(3.2, 4.0, "the same cell at every step, plus one new wire:\nthe state $\\mathbf{h}$ carried from step to step", ha="center", fontsize=11, color=S.OUTC)
    b.text(3.2, -0.65, "each output sees everything before it, through $\\mathbf{h}$", ha="center", fontsize=11, color=S.GREY)
    b.set_xlim(-1.0, 7.4); b.set_ylim(-1.3, 4.8); b.set_aspect("equal"); b.axis("off")
    b.set_title("(b) recurrence: sharing across time", loc="left")
    fig.tight_layout(w_pad=1.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
