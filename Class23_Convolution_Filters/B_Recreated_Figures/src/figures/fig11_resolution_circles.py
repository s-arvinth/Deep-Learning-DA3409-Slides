"""Three ways down, three ways back up, on a small grid of numbers.

The layout of Prince (2023), Figs. 10.11 and 10.12, drawn for this deck:
a 4 x 4 array of integers, each 2 x 2 block in its own tint, reduced to
2 x 2 and then restored to 4 x 4.  Every number shown is computed from
the array by the operation named, and the tints follow the numbers, so
the picture can be read as a calculation.

Down (top row):
  (a) sub-sampling keeps the top-left entry of each block
  (b) max pooling keeps the largest entry
  (c) mean pooling keeps the (rounded) average
Up (bottom row), from the max-pooled array:
  (d) duplication copies each entry into its block
  (e) max unpooling puts each entry back where the maximum was
      and fills the rest with zeros
  (f) bilinear interpolation, with zeros beyond the edge, blends
      neighbouring entries -- and the tints blend with them
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import to_rgb
from common import style as S

NAME = "resolution_circles"

A = np.array([[1, 3, 5, 3],
              [6, 2, 0, 8],
              [4, 6, 1, 4],
              [2, 8, 0, 3]])
TINTS = [[to_rgb("#D3D3D3"), to_rgb("#B9D9E3")],      # grey, teal tint
         [to_rgb("#9FB6C9"), to_rgb("#E3A995")]]      # indigo tint, maroon tint
R = 0.36
GAP = 0.85


def _tint_grid(n):
    """The block tint of each cell of an n x n grid."""
    T = np.empty((n, n, 3))
    for i in range(n):
        for j in range(n):
            T[i, j] = TINTS[i * 2 // n][j * 2 // n]
    return T


def _draw(ax, x0, y0, vals, tints, filled, dashed=None, div=True):
    n = vals.shape[0]
    for i in range(n):
        for j in range(n):
            cx, cy = x0 + GAP * j, y0 - GAP * i
            fc = tints[i, j] if filled[i, j] else "white"
            ls = (0, (2, 2)) if dashed is not None and dashed[i, j] else "-"
            ax.add_patch(Circle((cx, cy), R, fc=fc, ec="black", lw=1.0,
                                ls=ls, zorder=3))
            ax.text(cx, cy, "%d" % vals[i, j], ha="center", va="center",
                    fontsize=13, zorder=4)
    if div and n % 2 == 0:
        mid_x = x0 + GAP * (n / 2 - 0.5)
        mid_y = y0 - GAP * (n / 2 - 0.5)
        ax.plot([mid_x, mid_x], [y0 - GAP * (n - 0.5), y0 + GAP * 0.5],
                color="black", lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.plot([x0 - GAP * 0.5, x0 + GAP * (n - 0.5)], [mid_y, mid_y],
                color="black", lw=0.9, ls=(0, (3, 2)), zorder=2)
    return x0 + GAP * (n - 1)


def _arrow(ax, x, y):
    ax.annotate("", xy=(x + 0.9, y), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))


def _pool(A, how):
    n = A.shape[0] // 2
    out = np.empty((n, n), int)
    which = np.zeros_like(A, bool)
    for i in range(n):
        for j in range(n):
            blk = A[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            if how == "sub":
                out[i, j] = blk[0, 0]; which[2 * i, 2 * j] = True
            elif how == "max":
                out[i, j] = blk.max()
                k = np.unravel_index(np.argmax(blk), blk.shape)
                which[2 * i + k[0], 2 * j + k[1]] = True
            else:
                out[i, j] = int(round(blk.mean()))
                which[2 * i:2 * i + 2, 2 * j:2 * j + 2] = True
    return out, which


def _bilinear(B):
    """2 x 2 -> 4 x 4 with zeros beyond the right and bottom edge, as in
    Prince's drawing: output row 2r is input row r, row 2r+1 is the
    average of rows r and r+1 (zero past the edge); likewise columns."""
    P = np.zeros((3, 3)); P[:2, :2] = B
    rows = np.array([P[0], (P[0] + P[1]) / 2, P[1], (P[1] + P[2]) / 2])
    out = np.empty((4, 4))
    for i in range(4):
        r = rows[i]
        out[i] = [r[0], (r[0] + r[1]) / 2, r[1], (r[1] + r[2]) / 2]
    return out


def build():
    S.use()
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 4.6))
    T4 = _tint_grid(4); T2 = _tint_grid(2)
    ones4 = np.ones((4, 4), bool); ones2 = np.ones((2, 2), bool)

    # ---------------- down
    for ax, how, title in zip(axes[0], ("sub", "max", "mean"),
                              ("(a) sub-sampling", "(b) max pooling",
                               "(c) mean pooling")):
        out, which = _pool(A, how)
        print("%s -> %s" % (title, out.tolist()))
        xe = _draw(ax, 0, 0, A, T4, which)
        _arrow(ax, xe + 0.6, -GAP * 1.5)
        _draw(ax, xe + 2.0, -GAP * 1.0, out, T2, ones2)
        ax.set_xlim(-0.6, xe + 3.6); ax.set_ylim(-GAP * 3.6, 0.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=14)

    B, which_max = _pool(A, "max")

    # ---------------- up
    # (d) duplication
    ax = axes[1, 0]
    D = np.kron(B, np.ones((2, 2), int))
    xe = _draw(ax, 0, -GAP * 1.0, B, T2, ones2)
    _arrow(ax, xe + 0.6, -GAP * 1.5)
    _draw(ax, xe + 2.0, 0, D, T4, ones4)
    ax.set_title("(d) duplication", loc="left", fontsize=14)
    # (e) max unpooling: back to the remembered positions
    ax = axes[1, 1]
    U = np.zeros((4, 4), int)
    for i in range(2):
        for j in range(2):
            blk = which_max[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            k = np.argwhere(blk)[0]
            U[2 * i + k[0], 2 * j + k[1]] = B[i, j]
    xe = _draw(ax, 0, -GAP * 1.0, B, T2, ones2)
    _arrow(ax, xe + 0.6, -GAP * 1.5)
    _draw(ax, xe + 2.0, 0, U, T4, U != 0)
    ax.set_title("(e) max unpooling", loc="left", fontsize=14)
    # (f) bilinear
    ax = axes[1, 2]
    Bp = np.zeros((3, 3), int); Bp[:2, :2] = B
    dashed = np.zeros((3, 3), bool); dashed[2, :] = True; dashed[:, 2] = True
    T3 = np.ones((3, 3, 3))
    T3[:2, :2] = T2
    xe = _draw(ax, 0, -GAP * 0.5, Bp, T3, ~dashed, dashed=dashed, div=False)
    F = _bilinear(B)
    Tw = np.ones((3, 3, 3)); Tw[:2, :2] = T2
    TF = np.empty((4, 4, 3))
    for c in range(3):
        TF[:, :, c] = _bilinear_tint(Tw[:, :, c])
    _arrow(ax, xe + 0.6, -GAP * 1.5)
    _draw(ax, xe + 2.0, 0, np.rint(F).astype(int), TF, ones4, div=False)
    print("bilinear -> %s" % np.rint(F).astype(int).tolist())
    ax.set_title("(f) bilinear interpolation", loc="left", fontsize=14)
    for ax in axes[1]:
        ax.set_xlim(-0.6, GAP * 1 + 3.6 + GAP * 3); ax.set_ylim(-GAP * 3.6, 0.6)
        ax.set_aspect("equal"); ax.axis("off")

    fig.tight_layout(w_pad=0.3, h_pad=0.3)
    S.save(fig, NAME)


def _bilinear_tint(P):
    """The same interpolation applied to a 3 x 3 padded tint channel,
    with white (1.0) beyond the edge."""
    rows = np.array([P[0], (P[0] + P[1]) / 2, P[1], (P[1] + P[2]) / 2])
    out = np.empty((4, 4))
    for i in range(4):
        r = rows[i]
        out[i] = [r[0], (r[0] + r[1]) / 2, r[1], (r[1] + r[2]) / 2]
    return out


if __name__ == "__main__":
    build()
