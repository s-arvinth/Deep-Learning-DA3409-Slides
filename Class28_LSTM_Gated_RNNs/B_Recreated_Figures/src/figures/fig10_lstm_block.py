"""The LSTM cell as a block diagram, in the layout of Goodfellow et al.
(2016), Fig. 10.16: an input unit whose value the input gate lets into
the state; the state's linear self-loop, whose weight is the forget
gate (the black square is a one-step delay); and the output gate that
lets the state out."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from common import style as S

NAME = "lstm_block"
GATE, CELL, ACT = S.OUTC, S.L2, S.L1
GATE_F, CELL_F, ACT_F = "#F1D6D6", "#D6E6EC", "#E4DDEB"


def unit(ax, x, y, color, fill, label, r=0.42, sig=True):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.4, zorder=4))
    if sig:
        t = np.linspace(-2.5, 2.5, 40); ax.plot(x + t * 0.11, y + 0.22 * np.tanh(t) , color=color, lw=1.4, zorder=5)
    ax.text(x, y - r - 0.28, label, ha="center", va="top", fontsize=11, color=color)


def op(ax, x, y, label, color, fill, r=0.34):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.4, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=13, color=color, zorder=5)


def arrow(ax, p, q, color="black", lw=1.3, rad=0.0, z=3):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=%.2f" % rad), zorder=z)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    xs = [0.0, 3.0, 6.0, 9.0]; yb = 0.0
    names = ["input\n$\\mathbf{g}_t = \\tanh(\\cdot)$", "input gate\n$\\mathbf{i}_t = \\sigma(\\cdot)$", "forget gate\n$\\mathbf{f}_t = \\sigma(\\cdot)$", "output gate\n$\\mathbf{o}_t = \\sigma(\\cdot)$"]
    cols = [ACT, GATE, GATE, GATE]; fills = [ACT_F, GATE_F, GATE_F, GATE_F]
    for x, n, c, f in zip(xs, names, cols, fills):
        ax.add_patch(Circle((x, yb), 0.42, fc=f, ec=c, lw=1.4, zorder=4))
        tt = np.linspace(-2.5, 2.5, 40); ax.plot(x + tt * 0.11, yb + 0.22 * np.tanh(tt), color=c, lw=1.4, zorder=5)
        arrow(ax, (x - 0.3, yb - 1.35), (x - 0.3, yb - 0.5), lw=1.6); arrow(ax, (x + 0.3, yb - 1.35), (x + 0.3, yb - 0.5), lw=1.6)
        ax.text(x - 0.42, yb - 1.5, "$\\mathbf{x}_t$", ha="center", va="top", fontsize=10.5)
        ax.text(x + 0.5, yb - 1.5, "$\\mathbf{h}_{t-1}$", ha="center", va="top", fontsize=10.5, color=ACT)
        ax.text(x, yb - 2.15, n, ha="center", va="top", fontsize=11, color=c)
    # input x input gate
    op(ax, 0.0, 2.2, "$\\times$", CELL, "white")
    arrow(ax, (0.0, 0.44), (0.0, 1.84), color=ACT, lw=1.6)
    arrow(ax, (3.0 - 0.25, 0.36), (0.38, 2.0), color=GATE, lw=1.6)
    ax.text(1.9, 1.35, "$\\mathbf{i}_t \\odot \\mathbf{g}_t$", ha="center", va="center", fontsize=10.5, color=CELL, bbox=dict(fc="white", ec="none", pad=1.5))
    # state
    op(ax, 0.0, 4.6, "$+$", CELL, CELL_F, r=0.44)
    ax.text(-0.6, 4.6, "state $\\mathbf{c}_t$", ha="right", va="center", fontsize=11.5, color=CELL)
    arrow(ax, (0.0, 2.56), (0.0, 4.14), color=CELL, lw=1.6)
    # self-loop: up from the state, across, down through the delay and the forget x, back in
    ax.plot([0.0, 0.0, 3.0, 3.0], [5.04, 6.0, 6.0, 5.0], color=CELL, lw=1.8, zorder=2)
    ax.add_patch(Rectangle((1.35, 5.85), 0.3, 0.3, fc="black", ec="black", zorder=5))
    ax.text(3.3, 6.05, "self-loop,\none-step delay", ha="left", va="center", fontsize=10.5, color=CELL)
    op(ax, 3.0, 4.6, "$\\times$", CELL, "white")
    arrow(ax, (3.0, 6.0), (3.0, 4.96), color=CELL, lw=1.8)
    arrow(ax, (2.64, 4.6), (0.46, 4.6), color=CELL, lw=1.8)
    ax.text(1.5, 4.85, "$\\mathbf{f}_t \\odot \\mathbf{c}_{t-1}$", ha="center", va="bottom", fontsize=10.5, color=CELL)
    arrow(ax, (6.0 - 0.15, 0.44), (3.3, 4.2), color=GATE, lw=1.6)
    # output: state -> tanh -> x with output gate -> h_t
    op(ax, 0.0, 7.4, "$\\times$", ACT, "white")
    arrow(ax, (0.0, 5.04), (0.0, 7.04), color=CELL, lw=1.6)
    ax.text(-0.35, 6.9, "$\\tanh(\\mathbf{c}_t)$", ha="right", va="center", fontsize=10.5, color=CELL)
    ax.annotate("", xy=(0.4, 7.25), xytext=(9.0 + 0.15, 0.44), arrowprops=dict(arrowstyle="-|>", color=GATE, lw=1.6, connectionstyle="arc3,rad=0.38", shrinkA=1, shrinkB=1), zorder=3)
    arrow(ax, (0.0, 7.76), (0.0, 8.8), color=ACT, lw=1.8)
    ax.text(0.35, 8.5, "output $\\mathbf{h}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t)$", ha="left", va="center", fontsize=11.5, color=ACT)
    # h_t fed back for the next step
    ax.plot([0.0, -1.6, -1.6], [8.2, 8.2, -1.9], color=ACT, lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.add_patch(Rectangle((-1.75, 3.0), 0.3, 0.3, fc="black", ec="black", zorder=5))
    ax.text(-1.95, 3.15, "$\\mathbf{h}_{t-1}$ for the\nnext step", ha="right", va="center", fontsize=10, color=ACT)
    ax.text(5.6, 7.6, "every gate is a sigmoid of $\\mathbf{x}_t$ and $\\mathbf{h}_{t-1}$: a number in $(0, 1)$\nper coordinate, multiplied into what it gates --- a soft switch\nthe network sets at every step",
            ha="left", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-4.6, 11.6); ax.set_ylim(-3.4, 9.2); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
