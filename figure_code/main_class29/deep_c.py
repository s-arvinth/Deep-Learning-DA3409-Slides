"""deep_c — deep RNN (c): the same with a skip connection (Goodfellow Fig. 10.13c layout)

Used in: Class 29 (B_Recreated_Figures), slide(s): "Three ways to make an RNN deep"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python deep_c.py            -> output/deep_c.pdf
         python deep_c.py --out DIR  -> DIR/deep_c.pdf

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/figures/fig12_deep_ways.py (build_c()),
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

NAME = "deep_c"
SRC, OUT = S.L1, S.OUTC


def node(ax, x, y, label, color, fill, r=0.4, fs=12):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.4, zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def arrow(ax, p, q, color="black", lw=1.4, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=%.2f" % rad), zorder=2)


def base(ax):
    ax.set_xlim(-1.6, 3.0); ax.set_ylim(-0.9, 5.7); ax.set_aspect("equal"); ax.axis("off")


def build():
    S.use(); fig, ax = plt.subplots(figsize=(4.4, 6.0))
    node(ax, 0, 0, "$\\mathbf{x}$", S.GREY, "#EEEEEE")
    node(ax, 0, 1.4, "", SRC, "white", r=0.36)
    node(ax, 0, 2.8, "$\\mathbf{h}$", SRC, "#E4DDEB")
    node(ax, 0, 4.1, "", OUT, "white", r=0.36)
    node(ax, 0, 5.2, "$\\mathbf{y}$", OUT, "#F1D6D6", r=0.36)
    arrow(ax, (0, 0.42), (0, 1.02)); arrow(ax, (0, 1.78), (0, 2.38)); arrow(ax, (0, 3.22), (0, 3.72)); arrow(ax, (0, 4.48), (0, 4.82))
    node(ax, 1.5, 1.5, "", SRC, "white", r=0.36)
    ax.annotate("", xy=(1.5, 1.88), xytext=(0.42, 2.8), arrowprops=dict(arrowstyle="-|>", color=SRC, lw=1.4, connectionstyle="arc3,rad=-0.4", shrinkA=0, shrinkB=0), zorder=2)
    ax.annotate("", xy=(0.4, 2.6), xytext=(1.2, 1.3), arrowprops=dict(arrowstyle="-|>", color=SRC, lw=1.4, connectionstyle="arc3,rad=-0.1", shrinkA=0, shrinkB=0), zorder=2)
    ax.add_patch(Rectangle((0.62, 1.95), 0.32, 0.32, fc="black", ec="black", zorder=5))
    # the skip connection: h -> delay -> h directly, a loop on the right
    ax.plot([0.3, 1.7, 1.7], [3.1, 3.1, 2.6], color=OUT, lw=1.6, zorder=2)
    arrow(ax, (1.7, 2.6), (0.42, 2.7), color=OUT, lw=1.6)
    ax.add_patch(Rectangle((1.54, 3.3), 0.32, 0.32, fc="black", ec="black", zorder=5))
    ax.plot([1.7, 1.7], [3.1, 3.3], color=OUT, lw=1.6, zorder=2)
    ax.text(2.0, 3.45, "skip", ha="left", va="center", fontsize=9, color=OUT)
    ax.text(1.55, 0.8, "the extra layer,\nand a direct path", ha="center", va="top", fontsize=9, color=S.GREY)
    base(ax); S.save(fig, "deep_c")


if __name__ == "__main__":
    build()
