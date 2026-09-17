"""Drawing helpers in the style of Bishop & Bishop (2024), Figs. 10.22,
10.23, 10.28-10.30: sheared grids seen in perspective, and flat grids
of numbered cells, in the book's four colours."""
import numpy as np
from matplotlib.patches import Polygon, Rectangle

RED, GREEN, BLUE, YELLOW, PURPLE = "#FF8080", "#80FF80", "#8080FF", "#FFFF80", "#C080C0"
WHITE = "white"
A, B, H = 0.40, 0.26, 0.62          # column step, rise per column, row height


def cell_corners(x0, y0, r, c):
    """Corners of cell (row r from the top, column c) of a sheared grid
    whose top-left corner is (x0, y0)."""
    x = x0 + c * A; y = y0 - r * H + c * B
    return [(x, y), (x + A, y + B), (x + A, y + B - H), (x, y - H)]


def draw_grid(ax, x0, y0, n, fills, default=YELLOW, lw=0.6, edge="black"):
    for r in range(n):
        for c in range(n):
            ax.add_patch(Polygon(cell_corners(x0, y0, r, c), closed=True,
                                 fc=fills.get((r, c), default), ec=edge, lw=lw,
                                 zorder=2))
    outer = [(x0, y0), (x0 + n * A, y0 + n * B), (x0 + n * A, y0 + n * B - n * H),
             (x0, y0 - n * H)]
    ax.add_patch(Polygon(outer, closed=True, fc="none", ec="black", lw=1.6, zorder=3))


def block_outline(ax, x0, y0, r0, c0, k, lw=1.6):
    """Heavy outline around a k x k block starting at (r0, c0); returns
    its four corners (tl, tr, br, bl)."""
    tl = cell_corners(x0, y0, r0, c0)[0]
    tr = cell_corners(x0, y0, r0, c0 + k - 1)[1]
    br = cell_corners(x0, y0, r0 + k - 1, c0 + k - 1)[2]
    bl = cell_corners(x0, y0, r0 + k - 1, c0)[3]
    ax.add_patch(Polygon([tl, tr, br, bl], closed=True, fc="none", ec="black",
                         lw=lw, zorder=4))
    return tl, tr, br, bl


def connect(ax, src, dst, color="black", lw=0.6):
    for (xa, ya), (xb, yb) in zip(src, dst):
        ax.plot([xa, xb], [ya, yb], color=color, lw=lw, zorder=1)


def flat_grid(ax, x0, y0, vals, colors, cell=1.0, fs=12, lw=1.0):
    """A flat grid of numbered square cells, top-left at (x0, y0)."""
    n0, n1 = vals.shape
    for i in range(n0):
        for j in range(n1):
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell,
                                   fc=colors[i][j], ec="black", lw=lw, zorder=2))
            ax.text(x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, "%d" % vals[i, j],
                    ha="center", va="center", fontsize=fs, zorder=3)
    ax.add_patch(Rectangle((x0, y0 - n0 * cell), n1 * cell, n0 * cell, fc="none",
                           ec="black", lw=2.0, zorder=3))
    return x0 + n1 * cell


def arrow(ax, x0, x1, y, lw=1.4):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=lw))
