"""A network with an explicit memory, in the layout of Goodfellow et al.
(2016), Fig. 10.18: a row of memory cells; a task network with its own
recurrence (the black square is a one-step delay) that controls a
writing mechanism and a reading mechanism; thick arrows carry the
content written and read, thin arrows the addresses the task network
chooses."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from common import style as S

NAME = "explicit_memory"


def arrow(ax, p, q, color="black", lw=1.2, rad=0.0, z=3):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=%.2f" % rad), zorder=z)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    x0, y0, cw, ch = 0.0, 4.8, 1.6, 1.0
    for k in range(5):
        ax.add_patch(Rectangle((x0 + k * cw, y0), cw, ch, fc="#D6E6EC", ec=S.L2, lw=1.4, zorder=3))
        ax.text(x0 + k * cw + cw / 2, y0 + ch / 2, "$\\mathbf{m}_{%d}$" % (k + 1), ha="center", va="center", fontsize=12, color=S.L2, zorder=4)
    ax.text(x0 + 2.5 * cw, y0 + ch + 0.25, "memory cells: each holds one vector, as an LSTM cell holds one $\\mathbf{c}_t$", ha="center", va="bottom", fontsize=11, color=S.L2)
    # task network
    ax.add_patch(FancyBboxPatch((1.6, 0.0), 4.8, 1.6, boxstyle="round,pad=0.05,rounding_size=0.4", fc="#E4DDEB", ec=S.L1, lw=1.5, zorder=3))
    ax.text(4.0, 0.8, "task network,\ncontrolling the memory", ha="center", va="center", fontsize=12, color=S.L1, zorder=4)
    ax.annotate("", xy=(1.7, 0.45), xytext=(1.7, 1.15), arrowprops=dict(arrowstyle="-|>", color=S.L1, lw=1.3, connectionstyle="arc3,rad=1.6", shrinkA=0, shrinkB=0), zorder=2)
    ax.add_patch(Rectangle((0.2, 0.65), 0.3, 0.3, fc="black", ec="black", zorder=5))
    ax.text(0.1, 0.3, "its own state,\none step back", ha="right", va="top", fontsize=9.5, color=S.L1)
    # writing: thick content arrow (left), thin address arrow (further left), labels on the far left
    arrow(ax, (2.6, 1.65), (1.6, 4.75), color=S.OUTC, lw=4.0)
    arrow(ax, (2.0, 1.65), (0.6, 4.75), color=S.OUTC, lw=0.9)
    ax.text(-0.3, 3.2, "writing mechanism", ha="right", va="center", fontsize=11, color=S.OUTC)
    ax.text(-0.3, 2.7, "thick: what is written;  thin: where", ha="right", va="center", fontsize=9.5, color=S.OUTC)
    # reading: thick content arrow down (right), delayed; thin address arrow up (further right); labels on the far right
    arrow(ax, (6.4, 4.75), (5.6, 1.65), color=S.ACC, lw=4.0)
    ax.add_patch(Rectangle((5.85, 3.05), 0.3, 0.3, fc="black", ec="black", zorder=5))
    arrow(ax, (6.2, 1.65), (7.4, 4.75), color=S.ACC, lw=0.9)
    ax.text(8.2, 3.2, "reading mechanism", ha="left", va="center", fontsize=11, color=S.ACC)
    ax.text(8.2, 2.7, "thick: what is read (one step later);  thin: where", ha="left", va="center", fontsize=9.5, color=S.ACC)
    ax.text(4.0, -0.9, "the addresses are a softmax over the cells --- Class 27's attention --- so writing and reading are differentiable",
            ha="center", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-5.4, 14.4); ax.set_ylim(-1.4, 6.6); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
