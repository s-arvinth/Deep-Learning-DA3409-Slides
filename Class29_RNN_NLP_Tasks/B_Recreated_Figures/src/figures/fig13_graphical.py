"""The RNN as a directed graphical model, in the layouts of Goodfellow
et al. (2016), Figs. 10.7 and 10.8.

`gm_complete`  the fully connected graph over y^(1), y^(2), ...: every
               past observation is a parent of every later one; the
               future, not yet observed, is dotted.
`gm_state`     the same with the state h^(t) in the graph: every stage
               has the same structure and can share parameters.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from common import style as S

NAME = "gm_complete"
NAMES = ("gm_complete", "gm_state")


def node(ax, x, y, label, color, fill, dotted=False, r=0.46, fs=11.5):
    ax.add_patch(Circle((x, y), r, fc=fill if not dotted else "white", ec=color, lw=1.3, ls=(0, (2, 2)) if dotted else "-", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def arrow(ax, p, q, color, lw=1.3, rad=0.0, dashed=False):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=(0, (4, 3)) if dashed else "-",
                                                    shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=%.2f" % rad), zorder=2)


def build_complete():
    S.use()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = [k * 1.9 for k in range(6)]
    for k, x in enumerate(xs):
        last = k == 5
        node(ax, x, 0, "$\\mathbf{y}^{(%s)}$" % ("\\ldots" if last else k + 1), S.OUTC, "#F1D6D6", dotted=last)
    for i in range(5):
        for j in range(i + 1, 6):
            dashed = j == 5
            if j == i + 1:
                arrow(ax, (xs[i] + 0.48, 0), (xs[j] - 0.48, 0), S.OUTC, dashed=dashed)
            else:
                arrow(ax, (xs[i] + 0.1, 0.46), (xs[j] - 0.1, 0.46), S.OUTC, lw=1.0, rad=-0.28 - 0.08 * (j - i), dashed=dashed)
    ax.set_xlim(-0.8, xs[-1] + 0.8); ax.set_ylim(-0.9, 4.2); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "gm_complete")


def build_state():
    S.use()
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    xs = [k * 1.9 for k in range(6)]
    for k, x in enumerate(xs):
        last = k == 5
        lab = "\\ldots" if last else str(k + 1)
        node(ax, x, 1.7, "$\\mathbf{h}^{(%s)}$" % lab, S.L1, "#E4DDEB", dotted=last)
        node(ax, x, 0, "$\\mathbf{y}^{(%s)}$" % lab, S.OUTC, "#F1D6D6", dotted=last)
        arrow(ax, (x, 1.22), (x, 0.48), S.L1, dashed=last)
        if k:
            arrow(ax, (xs[k - 1] + 0.48, 1.7), (x - 0.48, 1.7), S.L1, dashed=last)
            arrow(ax, (xs[k - 1] + 0.38, 0.3), (x - 0.38, 1.4), S.OUTC, dashed=last)
    ax.set_xlim(-0.8, xs[-1] + 0.8); ax.set_ylim(-0.7, 2.5); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "gm_state")


def build():
    build_complete(); build_state()


if __name__ == "__main__":
    build()
