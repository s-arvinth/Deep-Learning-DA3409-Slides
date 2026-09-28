"""Drawing helpers for unrolled networks in the layout of Jurafsky &
Martin's Figs. 14.17-14.19: a row of embedding boxes, a row of hidden
boxes, a row of softmax boxes, tokens below and outputs above, with
the encoder in indigo and the decoder in maroon."""
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
from . import style as S

ENC, DEC = S.L1, S.OUTC
ENC_F, DEC_F = "#E4DDEB", "#F1D6D6"


def column(ax, x, y0, color, fill, rows=("emb", "hid", "soft"), dy=0.9, w=0.6, h=0.5):
    """Draw the boxes of one time step; returns their centres by row."""
    cs = {}
    for k, r in enumerate(rows):
        y = y0 + k * dy
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fill, ec=color, lw=1.2, zorder=3))
        cs[r] = (x, y)
    return cs


def arrow(ax, p, q, color="black", lw=1.0, ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                    shrinkA=2, shrinkB=2), zorder=4)


def band(ax, x0, x1, y0, y1, color, label=None, alpha=0.10):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.15,rounding_size=0.25",
                                fc=color, ec="none", alpha=alpha, zorder=1))
    if label:
        ax.text((x0 + x1) / 2, y1 + 0.32, label, ha="center", va="bottom", color=color, fontsize=12)
