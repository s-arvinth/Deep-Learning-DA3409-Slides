"""Sequence classification, in the layouts of Jurafsky & Martin (2026),
Figs. 14.8 and 14.12.

`seq_classification`   one RNN reads x_1..x_n; its final state h_n goes
                       through a feedforward network and a softmax to
                       the class.
`bidir_classification` two RNNs, one in each direction; the final
                       forward state h_n^f and the final backward state
                       h_1^b are concatenated as the input to the same
                       feedforward classifier.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from common import style as S
from common import bookdraw as B

NAME = "seq_classification"
NAMES = ("seq_classification", "bidir_classification")


def pill(ax, x, y, label, color="black", w=0.42, h=0.7, fs=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", fc="white", ec=color, lw=1.0, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def classifier(ax, x, y0, label_in):
    """h -> FFN (trapezoid) -> softmax, stacked above (x, y0)."""
    pill(ax, x, y0, label_in, B.TGT, w=1.2, h=0.4, fs=10)
    ax.add_patch(Polygon([(x - 0.75, y0 + 0.35), (x + 0.75, y0 + 0.35), (x + 0.45, y0 + 1.2), (x - 0.45, y0 + 1.2)], closed=True, fc=B.TGT_F, ec=B.TGT, lw=1.2, zorder=3))
    ax.text(x, y0 + 0.78, "FFN", ha="center", va="center", fontsize=11.5, color=B.TGT, zorder=4)
    ax.add_patch(FancyBboxPatch((x - 0.75, y0 + 1.3), 1.5, 0.45, boxstyle="round,pad=0.02,rounding_size=0.2", fc="white", ec=B.TGT, lw=1.1, zorder=3))
    ax.text(x, y0 + 1.52, "softmax", ha="center", va="center", fontsize=10.5, color=B.TGT, zorder=4)
    B.arrow(ax, (x, y0 + 1.77), (x, y0 + 2.25), color=B.TGT, lw=1.2)
    ax.text(x, y0 + 2.45, "$\\hat{\\mathbf{y}}$", ha="center", va="center", fontsize=12, color=B.TGT)


def build_single():
    S.use()
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    xs = [0, 1.9, 3.8, 7.0]; labs = ["1", "2", "3", "n"]
    B.band(ax, -0.7, xs[-1] + 0.7, 0.0, h=1.0, color=B.SRC, label="RNN", lx=xs[-1] + 0.95)
    for k, (x, l) in enumerate(zip(xs, labs)):
        B.hidden(ax, x, 0.0, color=B.SRC, w=0.42)
        pill(ax, x, -1.3, "$\\mathbf{x}_{%s}$" % l)
        B.arrow(ax, (x, -0.94), (x, -0.23), lw=1.0)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.22, 0), (x - 0.22, 0), color=B.SRC, lw=1.3)
    ax.text(5.4, -1.3, "$\\cdots$", ha="center", va="center", fontsize=13, color=S.GREY)
    B.arrow(ax, (xs[-1], 0.23), (xs[-1], 0.85), color=B.SRC, lw=1.2)
    classifier(ax, xs[-1], 1.05, "$\\mathbf{h}_n$")
    ax.text(-0.7, 2.2, "no outputs at the earlier steps, so no loss there:\nthe one error signal is the classification's, and it is\nback-propagated through the classifier, then through\nthe RNN --- end-to-end training", ha="left", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-1.2, 10.0); ax.set_ylim(-1.9, 3.8); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "seq_classification")


def build_bidir():
    S.use()
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    xs = [0, 1.9, 3.8, 7.0]; labs = ["1", "2", "3", "n"]
    B.band(ax, -0.7, xs[-1] + 0.7, 0.0, h=1.0, color=B.SRC, label="RNN 1  ($\\rightarrow$)", lx=xs[-1] + 0.95)
    B.band(ax, -0.7, xs[-1] + 0.7, 1.7, h=1.0, color=S.L2, label="RNN 2  ($\\leftarrow$)", lx=xs[-1] + 0.95)
    for k, (x, l) in enumerate(zip(xs, labs)):
        B.hidden(ax, x, 0.0, color=B.SRC, w=0.42, fill=B.SRC_F if k == 3 else None, label="$\\mathbf{h}^f_n$" if k == 3 else None, fs=8)
        B.hidden(ax, x, 1.7, color=S.L2, w=0.42, fill="#D6E6EC" if k == 0 else None, label="$\\mathbf{h}^b_1$" if k == 0 else None, fs=8)
        pill(ax, x, -1.3, "$\\mathbf{x}_{%s}$" % l)
        B.arrow(ax, (x - 0.08, -0.94), (x - 0.08, -0.23), lw=1.0)
        B.arrow(ax, (x + 0.12, -0.94), (x + 0.12, 1.47), color=S.L2, lw=0.8)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.22, 0), (x - 0.22, 0), color=B.SRC, lw=1.3)
            B.arrow(ax, (x - 0.22, 1.7), (xs[k - 1] + 0.22, 1.7), color=S.L2, lw=1.3)
    ax.text(5.4, -1.3, "$\\cdots$", ha="center", va="center", fontsize=13, color=S.GREY)
    # the two final states into the classifier
    cx = 5.4
    ax.plot([xs[-1], xs[-1], cx + 0.3], [0.23, 0.9, 0.9], color=B.SRC, lw=1.2, zorder=2); B.arrow(ax, (cx + 0.3, 0.9), (cx + 0.3, 2.85), color=B.SRC, lw=1.2)
    ax.plot([xs[0], xs[0], cx - 0.3], [1.93, 2.6, 2.6], color=S.L2, lw=1.2, zorder=2); B.arrow(ax, (cx - 0.3, 2.6), (cx - 0.3, 2.85), color=S.L2, lw=1.2)
    ax.text(xs[-1] + 0.35, 0.6, "$\\mathbf{h}^f_n$", ha="left", va="center", fontsize=10, color=B.SRC)
    ax.text(xs[0] + 0.35, 2.45, "$\\mathbf{h}^b_1$", ha="left", va="center", fontsize=10, color=S.L2)
    classifier(ax, cx, 3.05, "$[\\mathbf{h}^f_n ; \\mathbf{h}^b_1]$")
    ax.text(-0.7, 4.4, "the final forward state has read the whole\ninput left to right, the final backward state\nright to left; concatenated, the classifier\nsees the beginning as well as the end", ha="left", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-1.2, 10.0); ax.set_ylim(-1.9, 5.9); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "bidir_classification")


def build():
    build_single(); build_bidir()


if __name__ == "__main__":
    build()
