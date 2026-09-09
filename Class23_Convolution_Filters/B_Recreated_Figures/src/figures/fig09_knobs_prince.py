"""Stride, kernel size and dilation, one output unit at a time.

The layout of Prince (2023), Fig. 10.3, in this deck's notation and
colours: the inputs x_1..x_8 in a column on the left, the outputs z_i on
the right, and between them the kernel weights w_j in boxes, each
multiplied (circled times) into the input it reads and summed (circled
plus) into the output.  Inputs that are read are shaded; outputs not
being computed are faded.

(a-b) kernel 3, stride 2: z_1 reads a zero pad and x_1, x_2; z_2 reads
      x_2, x_3, x_4.  Every other position is evaluated.
(c)   kernel 5, stride 1: z_4 reads x_2..x_6.  More reach, five weights.
(d)   kernel 3, dilation 2: z_5 reads x_3, x_5, x_7.  The reach of a
      kernel of five for the price of three.

The inputs each output reads are taken from `conv_matrix`, so the
picture is the layer the code actually applies.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from common import style as S
from common.models import conv_matrix, out_size

NAME = "conv_knobs"

N = 8
R = 0.36                 # unit radius: room for a letter and a subscript
RO = 0.17                # operator radius
XM, XW, XP, XO = 0.95, 1.95, 3.05, 3.65   # x of  (x), box, (+), z
BW, BH = 0.80, 0.56      # weight box
CASES = (  # (k, s, d, p, which output, title)
    (3, 2, 1, 1, 0, "(a) size 3, stride 2"),
    (3, 2, 1, 1, 1, "(b) size 3, stride 2"),
    (5, 1, 1, 2, 3, "(c) size 5, stride 1"),
    (3, 1, 2, 2, 4, "(d) size 3, dilation 2"),
)


def _circle(ax, xy, text, fill, faded=False):
    ec = S.GREY if faded else "black"
    fc = "#D9D9D9" if fill else "white"
    ax.add_patch(Circle(xy, R, fc=fc, ec=ec, lw=1.1, zorder=5,
                        alpha=0.45 if faded else 1.0))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=13,
            zorder=6, color=S.GREY if faded else "black")


def _op(ax, xy, sym):
    ax.add_patch(Circle(xy, RO, fc="white", ec=S.OUTC, lw=1.0, zorder=6))
    ax.text(xy[0], xy[1], sym, ha="center", va="center", fontsize=10,
            color=S.OUTC, zorder=7)


def _weight_row(ax, y, t, x_from):
    """One input (or the pad) at height y, multiplied by w_t and sent
    to the summation bus."""
    ax.plot([x_from, XW - BW / 2], [y, y], color=S.OUTC, lw=1.3, zorder=3)
    _op(ax, (XM, y), r"$\times$")
    ax.add_patch(Rectangle((XW - BW / 2, y - BH / 2), BW, BH, fc="white",
                           ec="black", lw=1.1, zorder=5))
    ax.text(XW, y, r"$w_{%d}$" % (t + 1), ha="center", va="center",
            fontsize=13, zorder=6)
    ax.plot([XW + BW / 2, XP - 0.45], [y, y], color=S.OUTC, lw=1.3, zorder=3)


def _panel(ax, k, s, d, p, i, title):
    M = conv_matrix(N, np.arange(1, k + 1), s=s, d=d, p=p)
    n_out = M.shape[0]
    assert n_out == out_size(N, k, s, d, p)
    reads = [(j, int(M[i, j]) - 1) for j in range(N) if M[i, j] != 0]
    pads = [(t, i * s + t * d - p) for t in range(k)
            if not 0 <= i * s + t * d - p < N]
    ys_in = np.arange(N - 1, -1, -1.0)
    ys_out = np.array([N - 1 - (i2 * s + (d * (k - 1)) / 2 - p)
                       for i2 in range(n_out)])
    ys_box = []
    for j, t in reads:
        _weight_row(ax, ys_in[j], t, R)
        ys_box.append(ys_in[j])
    for t, pos in pads:
        y = N - 1 - pos
        ax.text(0.0, y, "0", ha="center", va="center", fontsize=13,
                color=S.OUTC, zorder=6)
        _weight_row(ax, y, t, 0.22)
        ys_box.append(y)
    yo = ys_out[i]
    ax.plot([XP - 0.45, XP - 0.45], [min(ys_box), max(ys_box)], color=S.OUTC,
            lw=1.3, zorder=3)
    ax.plot([XP - 0.45, XO - R], [yo, yo], color=S.OUTC, lw=1.3, zorder=3)
    _op(ax, (XP, yo), "+")
    for j in range(N):
        _circle(ax, (0, ys_in[j]), r"$x_{%d}$" % (j + 1),
                fill=any(j == jj for jj, _ in reads))
    for i2 in range(n_out):
        _circle(ax, (XO, ys_out[i2]), r"$z_{%d}$" % (i2 + 1), fill=(i2 == i),
                faded=(i2 != i))
    ax.text(XO / 2, -1.0, "$k = %d$,  $s = %d$,  $d = %d$" % (k, s, d),
            ha="center", va="top", fontsize=12, color=S.GREY)
    ax.set_xlim(-R - 0.25, XO + R + 0.25); ax.set_ylim(-1.9, N + 0.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=14, pad=8)


def build():
    S.use()
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 4.4))
    for ax, (k, s, d, p, i, title) in zip(axes, CASES):
        _panel(ax, k, s, d, p, i, title)
    fig.tight_layout(w_pad=0.2)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
