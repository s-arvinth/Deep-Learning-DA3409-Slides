"""deep_a — deep RNN (a): hierarchical hidden-state groups (Goodfellow Fig. 10.13a layout)

Used in: Class 29 (B_Recreated_Figures), slide(s): "Three ways to make an RNN deep"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python deep_a.py            -> output/deep_a.pdf
         python deep_a.py --out DIR  -> DIR/deep_a.pdf

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/figures/fig12_deep_ways.py (build_a()),
same code, data, parameters and seeds.  The original module's description:

    Three ways to make a recurrent network deep, in the layout of
    Goodfellow et al. (2016), Fig. 10.13, one figure per panel: (a) the
    hidden state broken into groups organised hierarchically; (b) deeper
    computation -- an MLP -- in the input-to-hidden, hidden-to-hidden and
    hidden-to-output parts; (c) the same with a skip connection across the
    hidden-to-hidden path.  The black square is a one-step delay.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import common as S

NAME = "deep_a"
SRC, OUT = S.L1, S.OUTC


def node(ax, x, y, label, color, fill, r=0.4, fs=12):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.4, zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def arrow(ax, p, q, color="black", lw=1.4, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=%.2f" % rad), zorder=2)


def delay_loop(ax, x, y, r=0.4, side=1.0):
    """A self-loop through a delay square, on the right of a node: out of
    the top-right, round, and back into the bottom-right."""
    ax.plot([x + r * 0.75, x + 1.25, x + 1.25], [y + 0.28, y + 0.28, y - 0.28], color=SRC, lw=1.4, zorder=2)
    arrow(ax, (x + 1.25, y - 0.28), (x + r * 0.8, y - 0.28), color=SRC)
    ax.add_patch(Rectangle((x + 1.09, y - 0.16), 0.32, 0.32, fc="black", ec="black", zorder=5))


def base(ax):
    ax.set_xlim(-1.6, 3.0); ax.set_ylim(-0.9, 5.7); ax.set_aspect("equal"); ax.axis("off")


def build():
    S.use(); fig, ax = plt.subplots(figsize=(4.4, 6.0))
    node(ax, 0, 0, "$\\mathbf{x}$", S.GREY, "#EEEEEE")
    node(ax, 0, 1.6, "$\\mathbf{h}$", SRC, "#E4DDEB"); delay_loop(ax, 0, 1.6)
    node(ax, 0, 3.2, "$\\mathbf{z}$", SRC, "#E4DDEB"); delay_loop(ax, 0, 3.2)
    node(ax, 0, 4.8, "$\\mathbf{y}$", OUT, "#F1D6D6")
    for y0, y1 in ((0, 1.6), (1.6, 3.2), (3.2, 4.8)):
        arrow(ax, (0, y0 + 0.42), (0, y1 - 0.42))
    ax.text(1.55, 1.6, "one-step\ndelay", ha="left", va="center", fontsize=9, color=S.GREY)
    base(ax); S.save(fig, "deep_a")


if __name__ == "__main__":
    build()
