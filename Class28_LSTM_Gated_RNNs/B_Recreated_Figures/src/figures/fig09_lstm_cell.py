"""The LSTM unit as a computation graph, in the layout of Jurafsky &
Martin (2026), Fig. 14.13: inputs c_{t-1}, h_{t-1}, x_t on the left; the
projections of h_{t-1} and x_t fanning into four sums; a sigmoid or tanh
on each; the forget, candidate, add and output vectors; the elementwise
products and the sum that make the new cell c_t; tanh and the output
gate that make the new hidden h_t.  Gates in maroon, the cell line in
teal, activations in indigo."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, RegularPolygon
from common import style as S

NAME = "lstm_cell"
GATE, CELL, ACT = S.OUTC, S.L2, S.L1
GATE_F, CELL_F, ACT_F = "#F1D6D6", "#D6E6EC", "#E4DDEB"


def vbox(ax, x, y, h, label, color="black", fill="white", w=0.42, fs=10, lw=1.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fill, ec=color, lw=lw, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=5, rotation=0)


def diamond(ax, x, y, label, color, fill, r=0.3, fs=11):
    ax.add_patch(RegularPolygon((x, y), 4, radius=r, orientation=0, fc=fill, ec=color, lw=1.2, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=5)


def circ(ax, x, y, label, color, fill, r=0.24, fs=10):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.2, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=5)


def arrow(ax, p, q, color="black", lw=1.1, z=3):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1), zorder=z)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(15.0, 6.6))
    ax.add_patch(FancyBboxPatch((-0.4, -0.6), 15.6, 6.9, boxstyle="round,pad=0.1,rounding_size=0.5", fc="#F0EDF3", ec=S.L1, lw=1.2, zorder=0))
    ax.text(13.6, 0.0, "LSTM", fontsize=20, color=S.L1, ha="center", va="center")
    # inputs on the left
    yc, yh, yx = 5.5, 3.4, 0.8
    ax.text(-1.05, yc, "$\\mathbf{c}_{t-1}$", ha="right", va="center", fontsize=13, color=CELL)
    ax.text(-1.05, yh, "$\\mathbf{h}_{t-1}$", ha="right", va="center", fontsize=13, color=ACT)
    ax.text(-1.05, yx, "$\\mathbf{x}_t$", ha="right", va="center", fontsize=13)
    vbox(ax, 0.1, yc, 0.6, "", CELL, CELL_F, w=0.32)
    vbox(ax, 0.1, yh, 1.5, "", ACT, ACT_F, w=0.32)
    vbox(ax, 0.1, yx, 1.5, "", "black", "white", w=0.32)
    for y in (yc, yh, yx):
        arrow(ax, (-0.95, y), (-0.1, y))
    # the four rows: sum, activation, vector, name
    ys = [4.55, 3.2, 1.85, 0.5]
    xs_sum, xa, xv = 3.0, 4.3, 5.7
    for k, y in enumerate(ys):
        for (src_y, col, alpha) in ((yh, ACT, 0.28), (yx, S.GREY, 0.20)):
            ax.add_patch(Polygon([(0.28, src_y - 0.7), (0.28, src_y + 0.7), (xs_sum - 0.28, y + 0.14), (xs_sum - 0.28, y - 0.14)],
                                 closed=True, fc=col, ec=col, lw=0.4, alpha=alpha, zorder=1))
        circ(ax, xs_sum, y, "$+$", "black", "white", r=0.24, fs=11)
    ax.text(1.55, 5.95, "$\\mathbf{U}_{\\bullet}\\mathbf{h}_{t-1}$ and $\\mathbf{W}_{\\bullet}\\mathbf{x}_t$: one projection pair per row", ha="center", va="bottom", fontsize=9.5, color=S.GREY)
    acts = [("$\\sigma$", GATE, GATE_F, "$\\mathbf{f}_t$", "forget gate"), ("$\\tanh$", ACT, ACT_F, "$\\mathbf{g}_t$", "candidate"),
            ("$\\sigma$", GATE, GATE_F, "$\\mathbf{i}_t$", "add gate"), ("$\\sigma$", GATE, GATE_F, "$\\mathbf{o}_t$", "output gate")]
    for y, (lab, col, fill, vec, name) in zip(ys, acts):
        arrow(ax, (xs_sum + 0.26, y), (xa - 0.36, y))
        diamond(ax, xa, y, lab, col, fill, r=0.38, fs=9.5)
        arrow(ax, (xa + 0.38, y), (xv - 0.2, y))
        vbox(ax, xv, y, 0.9, vec, col, "white", w=0.4, fs=10.5)
        ax.text(xv, y - 0.62, name, ha="center", va="top", fontsize=9, color=col)
    # the cell line: c_{t-1} * f
    xm1, xm2, xplus, xc = 7.6, 7.6, 9.4, 10.6
    arrow(ax, (0.3, yc), (xm1 - 0.32, yc), color=CELL, lw=1.4)
    diamond(ax, xm1, yc, "$\\odot$", CELL, CELL_F, r=0.32)
    arrow(ax, (xv + 0.22, ys[0]), (xm1 - 0.22, yc - 0.26), color=GATE, lw=1.2)
    # g * i, on a row between g and i
    ym2 = 2.55
    diamond(ax, xm2, ym2, "$\\odot$", CELL, CELL_F, r=0.32)
    arrow(ax, (xv + 0.22, ys[1]), (xm2 - 0.26, ym2 + 0.18), color=ACT, lw=1.2)
    arrow(ax, (xv + 0.22, ys[2]), (xm2 - 0.26, ym2 - 0.18), color=GATE, lw=1.2)
    # sum
    yplus = yc - 0.55
    diamond(ax, xplus, yplus, "$+$", CELL, CELL_F, r=0.32)
    arrow(ax, (xm1 + 0.32, yc), (xplus - 0.24, yplus + 0.26), color=CELL, lw=1.4)
    arrow(ax, (xm2 + 0.32, ym2), (xplus - 0.24, yplus - 0.26), color=CELL, lw=1.4)
    # c_t
    vbox(ax, xc, yplus, 0.9, "$\\mathbf{c}_t$", CELL, CELL_F, w=0.4, fs=10.5)
    arrow(ax, (xplus + 0.32, yplus), (xc - 0.22, yplus), color=CELL, lw=1.4)
    arrow(ax, (xc + 0.22, yplus), (14.5, yplus), color=CELL, lw=1.6)
    ax.text(14.65, yplus, "$\\mathbf{c}_t$", ha="left", va="center", fontsize=13, color=CELL)
    # tanh, output gate, h_t
    xt, xm3, xh = 11.7, 12.9, 13.9
    yt, yh1 = 3.5, 2.6
    diamond(ax, xt, yt, "$\\tanh$", ACT, ACT_F, r=0.4, fs=8.5)
    arrow(ax, (xc, yplus - 0.47), (xt - 0.24, yt + 0.3), color=CELL, lw=1.2)
    diamond(ax, xm3, yh1, "$\\odot$", ACT, ACT_F, r=0.32)
    arrow(ax, (xt + 0.34, yt - 0.22), (xm3 - 0.26, yh1 + 0.2), color=ACT, lw=1.2)
    arrow(ax, (xv + 0.22, ys[3]), (xm3 - 0.26, yh1 - 0.2), color=GATE, lw=1.2)
    vbox(ax, xh, yh1, 0.9, "$\\mathbf{h}_t$", ACT, ACT_F, w=0.4, fs=10.5)
    arrow(ax, (xm3 + 0.32, yh1), (xh - 0.22, yh1), color=ACT, lw=1.2)
    arrow(ax, (xh + 0.22, yh1), (14.5, yh1), color=ACT, lw=1.6)
    ax.text(14.65, yh1, "$\\mathbf{h}_t$", ha="left", va="center", fontsize=13, color=ACT)
    ax.set_xlim(-2.0, 15.6); ax.set_ylim(-0.8, 6.6); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
