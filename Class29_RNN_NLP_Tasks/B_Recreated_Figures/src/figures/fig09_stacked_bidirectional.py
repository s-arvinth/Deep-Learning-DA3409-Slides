"""Stacked and bidirectional RNNs, in the layouts of Jurafsky & Martin
(2026), Figs. 14.10 and 14.11, drawn in the book's band style."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import style as S
from common import bookdraw as B

NAME = "stacked_bidirectional"


def pill(ax, x, y, label, color="black", w=0.42, h=0.7, fs=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", fc="white", ec=color, lw=1.0, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def build():
    S.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    xs = [0, 1.5, 3.0, 5.5]; labs = ["1", "2", "3", "n"]
    # (a) stacked
    bands = [(0.0, B.SRC, "RNN 1"), (1.4, S.L2, "RNN 2"), (2.8, B.TGT, "RNN 3")]
    for y, col, lab in bands:
        B.band(a, -0.6, xs[-1] + 0.6, y, h=0.9, color=col, label=lab, lx=xs[-1] + 0.85)
    for k, (x, l) in enumerate(zip(xs, labs)):
        for y, col, _ in bands:
            B.hidden(a, x, y, color=col, w=0.4)
        pill(a, x, -1.3, "$\\mathbf{x}_{%s}$" % l)
        B.arrow(a, (x, -0.94), (x, -0.22), lw=1.0)
        for (y0, _, _), (y1, _, _) in zip(bands[:-1], bands[1:]):
            B.arrow(a, (x, y0 + 0.22), (x, y1 - 0.22), lw=1.0)
        B.arrow(a, (x, 2.8 + 0.22), (x, 3.85), color=B.TGT, lw=1.0)
        pill(a, x, 4.2, "$\\mathbf{y}_{%s}$" % l, B.TGT)
        if k:
            for y, col, _ in bands:
                B.arrow(a, (xs[k - 1] + 0.21, y), (x - 0.21, y), color=col, lw=1.2)
    a.text(4.25, -1.3, "$\\cdots$", ha="center", va="center", fontsize=13, color=S.GREY)
    a.set_xlim(-1.0, 8.2); a.set_ylim(-1.9, 4.8); a.set_aspect("equal"); a.axis("off")
    a.set_title("(a) stacked: the output sequence of one layer is the input of the next", loc="left", fontsize=11.5)
    # (b) bidirectional
    B.band(b, -0.6, xs[-1] + 0.6, 0.0, h=0.9, color=B.SRC, label="RNN 1  ($\\rightarrow$)", lx=xs[-1] + 0.85)
    B.band(b, -0.6, xs[-1] + 0.6, 1.6, h=0.9, color=S.L2, label="RNN 2  ($\\leftarrow$)", lx=xs[-1] + 0.85)
    for k, (x, l) in enumerate(zip(xs, labs)):
        B.hidden(b, x, 0.0, color=B.SRC, w=0.4); B.hidden(b, x, 1.6, color=S.L2, w=0.4)
        pill(b, x, -1.3, "$\\mathbf{x}_{%s}$" % l)
        B.arrow(b, (x - 0.08, -0.94), (x - 0.08, -0.22), lw=1.0)
        B.arrow(b, (x + 0.12, -0.94), (x + 0.12, 1.38), color=S.L2, lw=0.8)
        # concatenated outputs: two small pills side by side, then y
        pill(b, x - 0.16, 3.0, "", B.SRC, w=0.26, h=0.5); pill(b, x + 0.16, 3.0, "", S.L2, w=0.26, h=0.5)
        B.arrow(b, (x - 0.16, 0.22), (x - 0.16, 2.74), color=B.SRC, lw=0.8)
        B.arrow(b, (x + 0.16, 1.82), (x + 0.16, 2.74), color=S.L2, lw=0.8)
        B.arrow(b, (x, 3.26), (x, 3.75), color=B.TGT, lw=1.0)
        pill(b, x, 4.1, "$\\mathbf{y}_{%s}$" % l, B.TGT)
        if k:
            B.arrow(b, (xs[k - 1] + 0.21, 0), (x - 0.21, 0), color=B.SRC, lw=1.2)
            B.arrow(b, (x - 0.21, 1.6), (xs[k - 1] + 0.21, 1.6), color=S.L2, lw=1.2)
    b.text(4.25, -1.3, "$\\cdots$", ha="center", va="center", fontsize=13, color=S.GREY)
    b.text(4.25, 3.0, "concatenated\noutputs $[\\mathbf{h}^f_t ; \\mathbf{h}^b_t]$", ha="center", va="center", fontsize=9.5, color=B.TGT)
    b.set_xlim(-1.0, 8.2); b.set_ylim(-1.9, 4.8); b.set_aspect("equal"); b.axis("off")
    b.set_title("(b) bidirectional: one network each way, states concatenated", loc="left", fontsize=11.5)
    fig.tight_layout(w_pad=0.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
