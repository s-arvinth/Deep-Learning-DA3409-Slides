"""Drawing helpers for the layouts of Jurafsky & Martin's Figs. 14.9 and
14.17-14.22: a row of embedding columns, a band of hidden-layer boxes,
a row of softmax boxes with a small histogram, words below and outputs
above.  Source side in indigo, target side in maroon, the context in
teal, attention in amber."""
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Ellipse
from . import style as S

SRC, TGT, CTX, ATT = S.L1, S.OUTC, S.L2, S.ACC
SRC_F, TGT_F, CTX_F, ATT_F = "#E4DDEB", "#F1D6D6", "#D6E6EC", "#F5EBD0"


def band(ax, x0, x1, y, h=0.9, color=SRC, fill=None, alpha=0.18, label=None, lx=None):
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0.05,rounding_size=0.3",
                                fc=fill or color, ec=color, lw=1.2, alpha=alpha if fill is None else 1.0, zorder=1))
    if label:
        ax.text(lx if lx is not None else x0 + 0.25, y, label, ha="left", va="center", fontsize=11, color=color, zorder=2)


def hidden(ax, x, y, color=SRC, fill=None, label=None, w=0.5, fs=8.5):
    ax.add_patch(Rectangle((x - w / 2, y - w / 2), w, w, fc=fill or "white", ec=color, lw=1.3, zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def embedding(ax, x, y, color=SRC, n=3, w=0.28, h=0.62):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec="black", lw=0.8, zorder=3))
    for k in range(n):
        yy = y - h / 2 + (k + 0.5) * h / n
        ax.add_patch(Circle((x, yy), 0.075, fc=[color, "#F0B0B0", "white"][k % 3] if False else color, ec=color, lw=0.5, alpha=0.35 + 0.3 * (k % 2), zorder=4))


def softmax(ax, x, y, color="black", w=0.62, h=0.36, bars=None, rng=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=color, lw=0.9, zorder=3))
    if bars is None:
        rng = rng or np.random.default_rng(0)
        bars = rng.random(7) ** 3; bars[rng.integers(7)] = 1.0
    bx = np.linspace(x - w / 2 + 0.08, x + w / 2 - 0.08, len(bars))
    for b, v in zip(bx, bars):
        ax.plot([b, b], [y - h / 2 + 0.05, y - h / 2 + 0.05 + 0.7 * h * v / max(bars)], color=color, lw=1.0, zorder=4)


def arrow(ax, p, q, color="black", lw=1.0, ls="-", rad=0.0, z=4):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1,
                                                    connectionstyle="arc3,rad=%.2f" % rad), zorder=z)


def brace(ax, x0, x1, y, label, color="black", up=False, fs=11):
    """A curly brace below (or above) a span, with a label."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    xm = (x0 + x1) / 2; d = 0.18 * (1 if up else -1)
    verts = [(x0, y), (x0, y + d), (xm - 0.15, y + d), (xm, y + 2 * d), (xm + 0.15, y + d), (x1, y + d), (x1, y)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO]
    ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=color, lw=1.0, zorder=3))
    ax.text(xm, y + 3.0 * d, label, ha="center", va="top" if not up else "bottom", fontsize=fs, color=color)


def column(ax, x, y_word, y_emb, y_hid, y_soft=None, y_out=None, word="", out="", color=SRC, hid_label=None, hid_fill=None, out_color=None, bars=None, rng=None):
    """One time step: word, embedding, hidden box, optional softmax and output."""
    ax.text(x, y_word, word, ha="center", va="center", fontsize=11.5, color=color)
    embedding(ax, x, y_emb, color)
    arrow(ax, (x, y_word + 0.22), (x, y_emb - 0.33), color="black", lw=0.9)
    hidden(ax, x, y_hid, color, fill=hid_fill, label=hid_label)
    arrow(ax, (x, y_emb + 0.33), (x, y_hid - 0.27), color="black", lw=0.9)
    if y_soft is not None:
        softmax(ax, x, y_soft, bars=bars, rng=rng)
        arrow(ax, (x, y_hid + 0.27), (x, y_soft - 0.2), color="black", lw=0.9)
    if y_out is not None:
        ax.text(x, y_out, out, ha="center", va="center", fontsize=11.5, color=out_color or color)
        arrow(ax, (x, y_soft + 0.2), (x, y_out - 0.22), color="black", lw=0.9)
