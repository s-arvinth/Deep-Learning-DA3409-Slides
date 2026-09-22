"""The Elman network, in the layouts of Jurafsky & Martin (2026), Figs.
14.1 and 14.2: (a) the three layers with the recurrent link from the
hidden layer back to its own input, dashed; (b) the same network drawn
as a feedforward step, the previous hidden layer multiplied by U and
added to the feedforward component W x_t."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from common import style as S
from common import draw as D

NAME_A = "elman_network"
NAME_B = "elman_feedforward"


def layer(ax, x, y, label, color, fill, w=0.7, h=2.4, n=4, vertical=True):
    if vertical:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.3", fc=fill, ec=color, lw=1.4, zorder=3))
        for k in range(n):
            yy = y - h / 2 + (k + 0.5) * h / n
            ax.add_patch(plt.Circle((x, yy), 0.16, fc="white", ec=color, lw=1.0, zorder=4))
        ax.text(x, y - h / 2 - 0.45, label, ha="center", va="center", fontsize=13, color=color)
    else:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fill, ec=color, lw=1.4, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=12, color=color, zorder=5)


def build_a():
    S.use()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    layer(ax, 0, 0, "$\\mathbf{x}_t$", S.GREY, "#EEEEEE", h=2.8, n=5)
    layer(ax, 3.0, 0, "$\\mathbf{h}_t$", D.ENC, D.ENC_F, h=2.0, n=3)
    layer(ax, 6.0, 0, "$\\mathbf{y}_t$", D.DEC, D.DEC_F, h=1.6, n=2)
    D.arrow(ax, (0.4, 0), (2.6, 0), color="black", lw=1.4); ax.text(1.5, 0.3, "$\\mathbf{W}$", ha="center", fontsize=12)
    D.arrow(ax, (3.4, 0), (5.6, 0), color="black", lw=1.4); ax.text(4.5, 0.3, "$\\mathbf{V}$", ha="center", fontsize=12)
    # the recurrent link: out of the top of h, round, back into its input side
    ax.plot([3.0, 3.0, 1.9, 1.9], [1.05, 1.9, 1.9, 0.55], color=D.ENC, lw=1.5, ls=(0, (4, 2)), zorder=2)
    ax.annotate("", xy=(2.62, 0.35), xytext=(1.9, 0.55), arrowprops=dict(arrowstyle="-|>", color=D.ENC, lw=1.5, ls=(0, (4, 2)), shrinkA=0, shrinkB=0), zorder=2)
    ax.text(2.45, 2.2, "$\\mathbf{U}$: the hidden layer's own value\nfrom the previous step, $\\mathbf{h}_{t-1}$", ha="center", va="bottom", fontsize=11, color=D.ENC)
    ax.set_xlim(-1.0, 7.2); ax.set_ylim(-2.3, 3.6); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME_A)


def build_b():
    S.use()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    # bottom row: h_{t-1} on the left, x_t on the right; weight slabs above them; sum; h_t; V; y_t
    layer(ax, 0.0, 0.0, "$\\mathbf{h}_{t-1}$", D.ENC, D.ENC_F, w=2.6, h=0.6, vertical=False)
    layer(ax, 5.0, 0.0, "$\\mathbf{x}_t$", S.GREY, "#EEEEEE", w=2.6, h=0.6, vertical=False)
    # slabs (parallelograms) for U and W
    ax.add_patch(Polygon([(-1.3, 0.55), (1.3, 0.55), (2.1, 1.75), (-0.5, 1.75)], closed=True, fc="#D6E6EC", ec=S.L2, lw=1.3, zorder=3))
    ax.text(0.4, 1.15, "$\\mathbf{U}$", ha="center", va="center", fontsize=13, color=S.L2, zorder=4)
    ax.add_patch(Polygon([(3.7, 0.55), (6.3, 0.55), (5.9, 1.75), (4.1, 1.75)], closed=True, fc="#D6E6EC", ec=S.L2, lw=1.3, zorder=3))
    ax.text(5.0, 1.15, "$\\mathbf{W}$", ha="center", va="center", fontsize=13, color=S.L2, zorder=4)
    # the sum
    ax.add_patch(plt.Circle((5.0, 2.55), 0.28, fc="white", ec="black", lw=1.2, zorder=4)); ax.text(5.0, 2.55, "$+$", ha="center", va="center", fontsize=13, zorder=5)
    D.arrow(ax, (5.0, 1.8), (5.0, 2.22)); D.arrow(ax, (1.6, 1.8), (4.7, 2.45), color=S.L2)
    ax.text(2.9, 2.45, "$\\mathbf{U}\\mathbf{h}_{t-1}$", ha="center", fontsize=11, color=S.L2)
    ax.text(5.45, 2.0, "$\\mathbf{W}\\mathbf{x}_t$", ha="left", fontsize=11, color=S.L2)
    layer(ax, 5.0, 3.5, "$\\mathbf{h}_t = g(\\cdot)$", D.ENC, D.ENC_F, w=2.6, h=0.6, vertical=False)
    D.arrow(ax, (5.0, 2.85), (5.0, 3.18))
    ax.add_patch(Polygon([(3.7, 3.85), (6.3, 3.85), (5.9, 4.85), (4.1, 4.85)], closed=True, fc="#D6E6EC", ec=S.L2, lw=1.3, zorder=3))
    ax.text(5.0, 4.35, "$\\mathbf{V}$", ha="center", va="center", fontsize=13, color=S.L2, zorder=4)
    layer(ax, 5.0, 5.5, "$\\mathbf{y}_t$", D.DEC, D.DEC_F, w=2.6, h=0.6, vertical=False)
    D.arrow(ax, (5.0, 4.9), (5.0, 5.18))
    ax.text(0.0, -0.75, "from the previous step", ha="center", fontsize=10.5, color=D.ENC)
    ax.text(5.0, -0.75, "the current input", ha="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-1.8, 7.3); ax.set_ylim(-1.2, 6.1); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME_B)


def build():
    build_a(); build_b()


if __name__ == "__main__":
    build()
