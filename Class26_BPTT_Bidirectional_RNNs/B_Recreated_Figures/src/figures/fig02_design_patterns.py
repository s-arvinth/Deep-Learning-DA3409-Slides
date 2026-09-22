"""Three design patterns, in the layouts of Goodfellow et al. (2016),
Figs. 10.3-10.5: hidden-to-hidden recurrence with an output at every
step; recurrence from the output only; hidden recurrence with a single
output at the end."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from common import style as S
from common import draw as D

NAME = "design_patterns"


def node(ax, x, y, label, color, fill, r=0.27):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.2, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=10.5, color=color, zorder=4)


def column(ax, x, s, out=True, loss=True):
    node(ax, x, 0, "$\\mathbf{x}_{%s}$" % s, S.GREY, "#EEEEEE"); node(ax, x, 1, "$\\mathbf{h}_{%s}$" % s, D.ENC, D.ENC_F)
    D.arrow(ax, (x, 0.29), (x, 0.71)); ax.text(x + 0.1, 0.5, "$\\mathbf{W}$", fontsize=8.5, va="center")
    if out:
        node(ax, x, 2, "$\\mathbf{o}_{%s}$" % s, D.DEC, D.DEC_F); D.arrow(ax, (x, 1.29), (x, 1.71)); ax.text(x + 0.1, 1.5, "$\\mathbf{V}$", fontsize=8.5, va="center")
        if loss:
            node(ax, x, 3, "$L_{%s}$" % s, S.ACC, "#F5EBD0"); D.arrow(ax, (x, 2.29), (x, 2.71))
            node(ax, x, 4, "$\\mathbf{y}_{%s}$" % s, S.GREY, "white"); D.arrow(ax, (x, 3.71), (x, 3.29))


def build():
    S.use()
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.4))
    steps = ["t-1", "t", "t+1"]; xs = [0, 1.3, 2.6]
    # (a) hidden recurrence, output every step
    ax = axs[0]
    for x, s in zip(xs, steps):
        column(ax, x, s)
    for a, b in zip(xs[:-1], xs[1:]):
        D.arrow(ax, (a + 0.29, 1), (b - 0.29, 1), color=D.ENC); ax.text((a + b) / 2, 1.13, "$\\mathbf{U}$", ha="center", fontsize=8.5, color=D.ENC)
    D.arrow(ax, (xs[-1] + 0.29, 1), (xs[-1] + 0.8, 1), color=D.ENC); D.arrow(ax, (xs[0] - 0.8, 1), (xs[0] - 0.29, 1), color=D.ENC)
    ax.text(1.3, -0.75, "$\\mathbf{h}_{t-1} \\to \\mathbf{h}_t$: the state carries everything", ha="center", fontsize=10.5, color=D.ENC)
    ax.set_title("(a) hidden-to-hidden recurrence", loc="left", fontsize=12.5)
    # (b) output recurrence
    ax = axs[1]
    for x, s in zip(xs, steps):
        column(ax, x, s)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(b - 0.2, 1.22), xytext=(a + 0.2, 1.8),
                    arrowprops=dict(arrowstyle="-|>", color=D.DEC, lw=1.0, shrinkA=0, shrinkB=0), zorder=4)
    ax.annotate("", xy=(xs[-1] + 0.85, 1.35), xytext=(xs[-1] + 0.2, 1.8), arrowprops=dict(arrowstyle="-|>", color=D.DEC, lw=1.0), zorder=4)
    ax.annotate("", xy=(xs[0] - 0.2, 1.22), xytext=(xs[0] - 0.85, 1.7), arrowprops=dict(arrowstyle="-|>", color=D.DEC, lw=1.0), zorder=4)
    ax.text(1.95, 1.62, "$\\mathbf{U}$", fontsize=8.5, color=D.DEC)
    ax.text(1.3, -0.75, "$\\mathbf{o}_{t-1} \\to \\mathbf{h}_t$: only the output reaches the next step", ha="center", fontsize=10.5, color=D.DEC)
    ax.set_title("(b) recurrence from the output", loc="left", fontsize=12.5)
    # (c) single output
    ax = axs[2]
    xs3 = [0, 1.0, 2.0, 3.0]; st3 = ["1", "2", "\\ldots", "\\tau"]
    for x, s in zip(xs3, st3):
        if s == "\\ldots":
            ax.text(x, 1, "$\\cdots$", ha="center", va="center", fontsize=14, color=D.ENC); ax.text(x, 0, "$\\cdots$", ha="center", va="center", fontsize=14, color=S.GREY)
            continue
        column(ax, x, s, out=(s == "\\tau"))
    for a, b in zip(xs3[:-1], xs3[1:]):
        D.arrow(ax, (a + 0.29, 1), (b - 0.29, 1), color=D.ENC); ax.text((a + b) / 2, 1.13, "$\\mathbf{U}$", ha="center", fontsize=8.5, color=D.ENC)
    ax.text(1.5, -0.75, "one output and one loss, after the whole sequence", ha="center", fontsize=10.5, color=D.DEC)
    ax.set_title("(c) a single output at the end", loc="left", fontsize=12.5)
    for ax in axs:
        ax.set_xlim(-1.1, 3.7); ax.set_ylim(-1.0, 4.4); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(w_pad=0.3)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
