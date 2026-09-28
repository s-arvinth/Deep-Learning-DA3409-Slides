"""Three ways to make a recurrent network deep, in the layout of
Goodfellow et al. (2016), Fig. 10.13, one figure per panel: (a) the
hidden state broken into groups organised hierarchically; (b) deeper
computation -- an MLP -- in the input-to-hidden, hidden-to-hidden and
hidden-to-output parts; (c) the same with a skip connection across the
hidden-to-hidden path.  The black square is a one-step delay."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from common import style as S

NAME = "deep_a"
NAMES = ("deep_a", "deep_b", "deep_c")
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


def build_a():
    S.use(); fig, ax = plt.subplots(figsize=(4.4, 6.0))
    node(ax, 0, 0, "$\\mathbf{x}$", S.GREY, "#EEEEEE")
    node(ax, 0, 1.6, "$\\mathbf{h}$", SRC, "#E4DDEB"); delay_loop(ax, 0, 1.6)
    node(ax, 0, 3.2, "$\\mathbf{z}$", SRC, "#E4DDEB"); delay_loop(ax, 0, 3.2)
    node(ax, 0, 4.8, "$\\mathbf{y}$", OUT, "#F1D6D6")
    for y0, y1 in ((0, 1.6), (1.6, 3.2), (3.2, 4.8)):
        arrow(ax, (0, y0 + 0.42), (0, y1 - 0.42))
    ax.text(1.55, 1.6, "one-step\ndelay", ha="left", va="center", fontsize=9, color=S.GREY)
    base(ax); S.save(fig, "deep_a")


def build_b():
    S.use(); fig, ax = plt.subplots(figsize=(4.4, 6.0))
    node(ax, 0, 0, "$\\mathbf{x}$", S.GREY, "#EEEEEE")
    node(ax, 0, 1.4, "", SRC, "white", r=0.36)
    node(ax, 0, 2.8, "$\\mathbf{h}$", SRC, "#E4DDEB")
    node(ax, 0, 4.1, "", OUT, "white", r=0.36)
    node(ax, 0, 5.2, "$\\mathbf{y}$", OUT, "#F1D6D6", r=0.36)
    arrow(ax, (0, 0.42), (0, 1.02)); arrow(ax, (0, 1.78), (0, 2.38)); arrow(ax, (0, 3.22), (0, 3.72)); arrow(ax, (0, 4.48), (0, 4.82))
    # the recurrence through an extra layer, with the delay
    node(ax, 1.5, 1.5, "", SRC, "white", r=0.36)
    ax.annotate("", xy=(1.5, 1.88), xytext=(0.42, 2.8), arrowprops=dict(arrowstyle="-|>", color=SRC, lw=1.4, connectionstyle="arc3,rad=-0.4", shrinkA=0, shrinkB=0), zorder=2)
    ax.annotate("", xy=(0.4, 2.6), xytext=(1.2, 1.3), arrowprops=dict(arrowstyle="-|>", color=SRC, lw=1.4, connectionstyle="arc3,rad=-0.1", shrinkA=0, shrinkB=0), zorder=2)
    ax.add_patch(Rectangle((0.62, 1.95), 0.32, 0.32, fc="black", ec="black", zorder=5))
    ax.text(1.55, 0.8, "an extra layer\non the way back", ha="center", va="top", fontsize=9, color=S.GREY)
    ax.text(0.5, 1.4, "$\\mathbf{h}\\to\\mathbf{h}$", ha="left", va="center", fontsize=8.5, color=SRC)
    base(ax); S.save(fig, "deep_b")


def build_c():
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


def build():
    build_a(); build_b(); build_c()


if __name__ == "__main__":
    build()
