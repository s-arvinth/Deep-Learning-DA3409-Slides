"""A recursive network, in the layout of Goodfellow et al. (2016), Fig.
10.14: the computational graph is a tree rather than a chain; every
leaf reads an input through V, every internal node combines its two
children through U (left) and W (right); the root gives the output o,
compared with the target y in the loss L."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from common import style as S

NAME = "recursive_tree"


def node(ax, x, y, label, color, fill, r=0.42, fs=12):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.4, zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def arrow(ax, p, q, color, label=None, side=-1, lw=1.4):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1), zorder=2)
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx + 0.3 * side, my + 0.05, label, ha="center", va="center", fontsize=11, color=color,
                bbox=dict(fc="white", ec="none", pad=1.2), zorder=5)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    xl = [0, 2.0, 4.4, 6.4]
    # leaves' inputs
    for k, x in enumerate(xl):
        node(ax, x, 0, "$\\mathbf{x}^{(%d)}$" % (k + 1), S.GREY, "#EEEEEE")
        node(ax, x, 1.7, "", S.L1, "#E4DDEB")
        arrow(ax, (x, 0.44), (x, 1.26), S.L2, "$\\mathbf{V}$", side=-1)
    # first level
    for (xa, xb, xp) in ((xl[0], xl[1], 1.0), (xl[2], xl[3], 5.4)):
        node(ax, xp, 3.5, "", S.L1, "#E4DDEB")
        arrow(ax, (xa + 0.2, 2.07), (xp - 0.2, 3.13), S.L1, "$\\mathbf{U}$", side=-1)
        arrow(ax, (xb - 0.2, 2.07), (xp + 0.2, 3.13), S.L1, "$\\mathbf{W}$", side=1)
    # root
    node(ax, 3.2, 5.3, "", S.L1, "#E4DDEB")
    arrow(ax, (1.0 + 0.25, 3.87), (3.2 - 0.25, 4.95), S.L1, "$\\mathbf{U}$", side=-1)
    arrow(ax, (5.4 - 0.25, 3.87), (3.2 + 0.25, 4.95), S.L1, "$\\mathbf{W}$", side=1)
    # output, target, loss
    node(ax, 3.2, 6.9, "$\\mathbf{o}$", S.OUTC, "#F1D6D6")
    arrow(ax, (3.2, 5.74), (3.2, 6.46), S.OUTC)
    node(ax, 5.2, 6.9, "$\\mathbf{y}$", S.GREY, "white")
    node(ax, 4.2, 8.4, "$L$", S.ACC, "#F5EBD0")
    arrow(ax, (3.5, 7.25), (3.95, 8.05), S.ACC); arrow(ax, (4.9, 7.25), (4.45, 8.05), S.ACC)
    ax.text(7.4, 4.5, "one set of weights $\\mathbf{U}, \\mathbf{V}, \\mathbf{W}$\nat every node of the tree;\na sequence of length $\\tau$ reaches\nthe root in $O(\\log\\tau)$ compositions", ha="left", va="center", fontsize=11, color=S.GREY)
    ax.set_xlim(-0.8, 11.6); ax.set_ylim(-0.6, 9.1); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
