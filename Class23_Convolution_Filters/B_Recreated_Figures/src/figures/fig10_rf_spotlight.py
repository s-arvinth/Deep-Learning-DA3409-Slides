"""How far back one unit can see, layer by layer.

The layout of Prince (2023), Fig. 10.6: each layer is a column of units
(three channels wide after the input), and a grey "spotlight" shows
which units of the previous layer reach the shaded ones.  Units inside
the receptive field of the final highlighted unit are grey; the rest
are in the deck's maroon tint.

Every field is COMPUTED, not drawn: a unit of a kernel-3 layer with
'same' padding reads rows s*j - 1, s*j, s*j + 1 of the layer below, and
the field is the union of those rows over the units already in it.  The
build asserts that the count so obtained equals the recursion
r_l = r_{l-1} + (k_l - 1) prod_{j<l} s_j before boundary clipping.

Three files:
    rf_spot_ab   (a) one layer: three inputs.  (b) two layers: five.
    rf_spot_c    (c) a third layer with stride two: seven.
    rf_spot_d    (d) a fourth: eleven of the twelve inputs.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from common import style as S
from common.models import rf_size

N_IN = 12
K = 3
CH = 3
FIELD_C = "#C9C9C9"
OUT_C = "#E4B8AE"          # maroon tint
UNIT = 0.5                 # spacing between unit rows


def _fields(strides, target):
    """Rows in each layer that reach `target` in the last one.

    Returns a list, layer 0 (input) first, of sorted row indices, and
    the row counts per layer.
    """
    sizes = [N_IN]
    for s in strides:
        sizes.append(-(-sizes[-1] // s))
    fields = [None] * (len(strides) + 1)
    fields[-1] = {target}
    for l in range(len(strides), 0, -1):
        s = strides[l - 1]
        rows = set()
        for j in fields[l]:
            for t in (-1, 0, 1):
                r = s * j + t
                if 0 <= r < sizes[l - 1]:
                    rows.add(r)
        fields[l - 1] = rows
    return [sorted(f) for f in fields], sizes


def _layer(ax, x0, rows, width_units, field, ytop):
    """Draw one layer as a rounded box of circles; return the x extent."""
    w = width_units * UNIT + 0.22
    h = rows * UNIT + 0.22
    ax.add_patch(FancyBboxPatch((x0, ytop - h), w, h,
                                boxstyle="round,pad=0,rounding_size=0.18",
                                fc="white", ec="black", lw=1.4, zorder=2))
    for r in range(rows):
        cy = ytop - 0.11 - UNIT * (r + 0.5)
        for c in range(width_units):
            cx = x0 + 0.11 + UNIT * (c + 0.5)
            ax.add_patch(Circle((cx, cy), 0.20,
                                fc=FIELD_C if r in field else OUT_C,
                                ec="black", lw=0.7, zorder=3))
    return x0, x0 + w


def _panel(ax, strides, target, title, labels=True, show_title=True):
    fields, sizes = _fields(strides, target)
    depth = len(strides)
    n_field = len(fields[0])
    r_formula = rf_size([K] * depth, strides)[-1]
    print("%s: field reaches %d input rows (recursion gives %d before "
          "clipping at the boundary)" % (title, n_field, r_formula))
    assert n_field == r_formula

    ytop = N_IN * UNIT / 2 + 0.3
    x = 0.0
    extents = []
    for l in range(depth + 1):
        width_units = 1 if l == 0 else CH
        rows = sizes[l]
        top_l = rows * UNIT / 2 + 0.11 + 0.11
        x0, x1 = _layer(ax, x, rows, width_units, set(fields[l]), top_l)
        extents.append((x0, x1, rows, top_l))
        x = x1 + 1.35
    # spotlights: from the field rows of layer l to those of layer l + 1
    for l in range(depth):
        x0a, x1a, rows_a, top_a = extents[l]
        x0b, x1b, rows_b, top_b = extents[l + 1]
        fa, fb = fields[l], fields[l + 1]
        ya_hi = top_a - 0.11 - UNIT * fa[0]
        ya_lo = top_a - 0.11 - UNIT * (fa[-1] + 1)
        yb_hi = top_b - 0.11 - UNIT * fb[0]
        yb_lo = top_b - 0.11 - UNIT * (fb[-1] + 1)
        ax.add_patch(Polygon([(x1a, ya_hi), (x0b, yb_hi), (x0b, yb_lo),
                              (x1a, ya_lo)], closed=True, fc="#BBBBBB",
                             ec="none", alpha=0.55, zorder=1))
    if labels:
        for l, (x0, x1, rows, top_l) in enumerate(extents):
            lab = r"input $\mathbf{x}$" if l == 0 else \
                  r"hidden layer $\mathbf{H}_{%d}$" % l
            ax.text((x0 + x1) / 2, -ytop - 0.15, lab, ha="center", va="top",
                    fontsize=11.5)
        for l in range(depth):
            x0a, x1a, _, _ = extents[l]
            x0b, _, _, _ = extents[l + 1]
            cin = 1 if l == 0 else CH
            txt = r"$\Omega$: $%d \times %d \times %d$" % (cin, K, CH)
            if strides[l] > 1:
                txt += "\nstride %d" % strides[l]
            ax.text((x1a + x0b) / 2, ytop + 0.05, txt, ha="center",
                    va="bottom", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-0.3, x - 1.35 + 0.3)
    ax.set_ylim(-ytop - 0.75, ytop + 0.72)
    ax.set_aspect("equal"); ax.axis("off")
    if show_title:
        ax.set_title(title, loc="left", fontsize=14, pad=10)


def build():
    S.use()
    # (a) and (b) on one figure
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.55]})
    _panel(axes[0], [1], 5, "(a) one layer: three inputs")
    _panel(axes[1], [1, 1], 5, "(b) two layers: five inputs")
    fig.tight_layout(w_pad=0.6)
    S.save(fig, "rf_spot_ab")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    _panel(ax, [1, 1, 2], 2, "(c) a third layer with stride two: seven inputs", show_title=False)
    fig.tight_layout()
    S.save(fig, "rf_spot_c")

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    _panel(ax, [1, 1, 2, 1], 3, "(d) a fourth layer: eleven of the twelve inputs", show_title=False)
    fig.tight_layout()
    S.save(fig, "rf_spot_d")


if __name__ == "__main__":
    build()
