"""Unfolding, in the layout of Goodfellow et al. (2016), Figs. 10.2-10.3:
left, the circuit with the one-step delay (black square); right, the
same network as an unfolded computational graph with U, W, V shared."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from common import style as S
from common import draw as D

NAME = "unfolding"


def node(ax, x, y, label, color, fill, r=0.3):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.3, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=12, color=color, zorder=4)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    F = {"x": (S.GREY, "#EEEEEE"), "h": (D.ENC, D.ENC_F), "o": (D.DEC, D.DEC_F), "L": (S.ACC, "#F5EBD0"), "y": (S.GREY, "white")}
    # ---- circuit
    x0 = 0.0
    for k, (lab, key) in enumerate((("$\\mathbf{x}$", "x"), ("$\\mathbf{h}$", "h"), ("$\\mathbf{o}$", "o"), ("$L$", "L"), ("$\\mathbf{y}$", "y"))):
        node(ax, x0, k * 1.1, lab, *F[key])
    for k, lab in ((0, "$\\mathbf{W}$"), (1, "$\\mathbf{V}$")):
        D.arrow(ax, (x0, k * 1.1 + 0.32), (x0, (k + 1) * 1.1 - 0.32), color="black")
        ax.text(x0 + 0.18, k * 1.1 + 0.55, lab, fontsize=11, color="black")
    D.arrow(ax, (x0, 2.2 + 0.32), (x0, 3.3 - 0.32)); D.arrow(ax, (x0, 4.4 - 0.32), (x0, 3.3 + 0.32))
    # the delay loop on h: out of the right side, round a box, back in
    ax.plot([x0 + 0.3, x0 + 1.0, x0 + 1.0, x0 + 1.0], [1.1 + 0.15, 1.1 + 0.15, 1.1 + 0.15, 1.1 - 0.15], color=D.ENC, lw=1.1, zorder=2)
    ax.plot([x0 + 1.0, x0 + 1.0], [1.1 + 0.15, 1.1 - 0.15], color=D.ENC, lw=1.1, zorder=2)
    D.arrow(ax, (x0 + 1.0, 1.1 - 0.15), (x0 + 0.3, 1.1 - 0.15), color=D.ENC)
    ax.add_patch(plt.Rectangle((x0 + 0.9, 1.1 - 0.1), 0.2, 0.2, fc="black", ec="black", zorder=4))
    ax.text(x0 + 1.18, 1.1, "$\\mathbf{U}$", fontsize=11, color=D.ENC, va="center")
    ax.text(x0 + 0.62, 1.42, "delay", fontsize=9, color=S.GREY, ha="center")
    ax.annotate("", xy=(3.0, 2.2), xytext=(1.9, 2.2), arrowprops=dict(arrowstyle="-|>", lw=1.4, color=S.GREY))
    ax.text(2.45, 2.45, "unfold", ha="center", fontsize=11, color=S.GREY)
    # ---- unfolded
    xs = [4.0, 5.5, 7.0, 8.5]; labs = ["t-1", "t", "t+1", "\\ldots"]
    prev = None
    for x, s in zip(xs, labs):
        if s == "\\ldots":
            ax.text(x, 1.1, "$\\cdots$", ha="center", va="center", fontsize=16, color=D.ENC)
            D.arrow(ax, (prev, 1.1), (x - 0.35, 1.1), color=D.ENC); ax.text((prev + x) / 2, 1.28, "$\\mathbf{U}$", ha="center", fontsize=10, color=D.ENC)
            break
        node(ax, x, 0.0, "$\\mathbf{x}_{%s}$" % s, *F["x"]); node(ax, x, 1.1, "$\\mathbf{h}_{%s}$" % s, *F["h"])
        node(ax, x, 2.2, "$\\mathbf{o}_{%s}$" % s, *F["o"]); node(ax, x, 3.3, "$L_{%s}$" % s, *F["L"]); node(ax, x, 4.4, "$\\mathbf{y}_{%s}$" % s, *F["y"])
        D.arrow(ax, (x, 0.32), (x, 0.78)); ax.text(x + 0.18, 0.5, "$\\mathbf{W}$", fontsize=10)
        D.arrow(ax, (x, 1.42), (x, 1.88)); ax.text(x + 0.18, 1.6, "$\\mathbf{V}$", fontsize=10)
        D.arrow(ax, (x, 2.52), (x, 2.98)); D.arrow(ax, (x, 4.08), (x, 3.62))
        if prev is None:
            ax.text(x - 0.9, 1.1, "$\\mathbf{h}_{\\ldots}$", ha="center", va="center", fontsize=12, color=D.ENC)
            D.arrow(ax, (x - 0.55, 1.1), (x - 0.32, 1.1), color=D.ENC)
        else:
            D.arrow(ax, (prev + 0.32, 1.1), (x - 0.32, 1.1), color=D.ENC); ax.text((prev + x) / 2, 1.28, "$\\mathbf{U}$", ha="center", fontsize=10, color=D.ENC)
        prev = x
    ax.text(6.25, -0.75, "the same $\\mathbf{U}, \\mathbf{W}, \\mathbf{V}$ at every step; the graph's size is the sequence length", ha="center", fontsize=11, color=S.GREY)
    ax.set_xlim(-1.4, 9.5); ax.set_ylim(-1.0, 4.9); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
