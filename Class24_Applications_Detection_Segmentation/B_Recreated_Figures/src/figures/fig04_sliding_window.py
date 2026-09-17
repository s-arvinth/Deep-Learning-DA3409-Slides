"""A sliding window is a convolution, so run it once.

The layout of Bishop & Bishop (2024), Figs. 10.22 and 10.23, redrawn:
the toy network -- a 3 x 3 convolution, non-overlapping 2 x 2 pooling
and one fully connected output -- first on the 6 x 6 window it was
trained on, then on an 8 x 8 image with every layer enlarged.

Every grid size is computed from the layer arithmetic (`window_cost`),
not typed in: the convolution map is n - 2, the pooled map (n - 2) // 2,
and the output side is the number of window positions.  In the second
panel the yellow cells are those the first window alone needs, the blue
ones the extra computation for the other three windows; the green and
red patches trace one unit's receptive field back through the layers,
as Bishop draws it.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import bishopdraw as D
from common.models import window_cost

NAME = "sliding_window"


def panel(ax, n_img, title):
    pos, naive, shared = window_cost(n_img)
    c_side = n_img - 2; p_side = c_side // 2; o_side = int(round(np.sqrt(pos)))
    c_old, p_old = 4, 2                      # the 6 x 6 window's maps
    print("%s: conv %d, pool %d, outputs %d x %d; naive %d, shared %d"
          % (title, c_side, p_side, o_side, o_side, naive, shared))

    def fills(n, old, extra):
        f = {}
        if n_img > 6:
            for r in range(n):
                for c in range(n):
                    if r >= old or c >= old:
                        f[(r, c)] = D.BLUE
        f.update(extra)
        return f

    # ---- layer positions
    xs = [0.0, 4.4, 7.8, 10.6]
    tops = [2.2, 1.4, 0.7, 0.3]
    # input: green 3 x 3 window at rows 1-3, cols 0-2
    g_in = {(r, c): D.GREEN for r in range(1, 4) for c in range(0, 3)}
    D.draw_grid(ax, xs[0], tops[0], n_img, fills(n_img, 6, g_in))
    win = D.block_outline(ax, xs[0], tops[0], 1, 0, 3)
    # conv: red 2 x 2 at rows 1-2, cols 0-1 with green cell at (1, 0)
    r_c = {(r, c): D.RED for r in range(1, 3) for c in range(0, 2)}
    r_c[(1, 0)] = D.GREEN
    D.draw_grid(ax, xs[1], tops[1], c_side, fills(c_side, c_old, r_c))
    g_cell = D.block_outline(ax, xs[1], tops[1], 1, 0, 1)
    r_blk = D.block_outline(ax, xs[1], tops[1], 1, 0, 2)
    D.connect(ax, win, g_cell)
    # pool: red cell at (0, 0), green the rest of the first window's block
    p_f = {(r, c): D.GREEN for r in range(p_old) for c in range(p_old)}
    p_f[(0, 0)] = D.RED
    D.draw_grid(ax, xs[2], tops[2], p_side, fills(p_side, p_old, p_f))
    r_cell = D.block_outline(ax, xs[2], tops[2], 0, 0, 1)
    g_blk = D.block_outline(ax, xs[2], tops[2], 0, 0, p_old)
    D.connect(ax, r_blk, r_cell)
    # output: green cell at (0, 0)
    D.draw_grid(ax, xs[3], tops[3], o_side, fills(o_side, 1, {(0, 0): D.GREEN}))
    o_cell = D.block_outline(ax, xs[3], tops[3], 0, 0, 1)
    D.connect(ax, g_blk, o_cell)

    labs = ["$%d \\times %d$ input image" % (n_img, n_img), "$3 \\times 3$ convolution",
            "$2 \\times 2$ pooling", "fully connected"]
    for x, t, n, lab in zip(xs, tops, (n_img, c_side, p_side, o_side), labs):
        ax.text(x + 0.1, t + n * D.B + 0.35, lab, fontsize=11, ha="left", va="bottom")
    ax.set_xlim(-0.3, 12.4); ax.set_ylim(-4.2, 5.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    panel(axes[0], 6, "(a) the network on the window it was trained on")
    panel(axes[1], 8, "(b) enlarged to an $8 \\times 8$ image: blue is the added work")
    fig.tight_layout(w_pad=1.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
