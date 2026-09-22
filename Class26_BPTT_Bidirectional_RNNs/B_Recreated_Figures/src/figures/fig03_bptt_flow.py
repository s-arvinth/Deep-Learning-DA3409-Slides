"""The unrolled network of Jurafsky & Martin (2026), Fig. 14.4, drawn
large, with both passes: forward, left to right, every h_t stored;
backward, right to left, each h_t receiving one term from its own loss
and one from the state after it (Goodfellow Eq. 10.20)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from common import style as S
from common import draw as D

NAME = "bptt_flow"


def box(ax, x, y, label, color, fill, w=0.9, h=0.5, fs=12):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=color, lw=1.3, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    xs = [0, 3.2, 6.4, 9.6]; ys = [0, 1.1, 2.2, 3.3]
    box(ax, -2.2, 0, "$\\mathbf{h}_0 = \\mathbf{0}$", D.ENC, D.ENC_F, w=1.3)
    prev = (-2.2, 0)
    for k, (x, y) in enumerate(zip(xs, ys)):
        t = k + 1
        box(ax, x, y - 1.3, "$\\mathbf{x}_{%d}$" % t, S.GREY, "#EEEEEE")
        box(ax, x, y, "$\\mathbf{h}_{%d}$" % t, D.ENC, D.ENC_F)
        box(ax, x, y + 1.3, "$\\hat{\\mathbf{y}}_{%d}$" % t, D.DEC, D.DEC_F)
        box(ax, x, y + 2.4, "$L_{%d}$" % t, S.ACC, "#F5EBD0", w=0.7, h=0.44)
        # forward, solid
        D.arrow(ax, (x - 0.14, y - 1.03), (x - 0.14, y - 0.28), lw=1.2); ax.text(x - 0.32, y - 0.66, "$\\mathbf{W}$", fontsize=10.5, ha="right", va="center")
        D.arrow(ax, (x - 0.14, y + 0.28), (x - 0.14, y + 1.02), lw=1.2); ax.text(x - 0.32, y + 0.65, "$\\mathbf{V}$", fontsize=10.5, ha="right", va="center")
        D.arrow(ax, (x - 0.14, y + 1.57), (x - 0.14, y + 2.15), lw=1.2)
        D.arrow(ax, (prev[0] + (0.68 if k == 0 else 0.48), prev[1] + 0.1), (x - 0.48, y + 0.1), color=D.ENC, lw=1.4)
        ax.text((prev[0] + x) / 2 - 0.25, (prev[1] + y) / 2 + 0.42, "$\\mathbf{U}$", fontsize=10.5, color=D.ENC, ha="center")
        # backward, dashed maroon
        ax.annotate("", xy=(x + 0.14, y + 0.3), xytext=(x + 0.14, y + 2.13),
                    arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.5, ls=(0, (4, 2)), shrinkA=0, shrinkB=0), zorder=5)
        if k > 0:
            px, py = prev
            ax.annotate("", xy=(px + 0.48, py - 0.12), xytext=(x - 0.48, y - 0.12),
                        arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.5, ls=(0, (4, 2)), shrinkA=0, shrinkB=0), zorder=5)
        prev = (x, y)
    # the two terms, written once, in the empty corners
    ax.text(10.6, 6.15, "$\\mathbf{V}^{\\top}\\nabla_{\\mathbf{o}_t}L$", fontsize=13, color=S.OUTC, va="center")
    ax.text(10.6, 5.7, "from the loss at the same step", fontsize=10.5, color=S.GREY, va="center")
    ax.text(10.6, 4.35, "$\\mathbf{U}^{\\top}\\mathrm{diag}(1-\\mathbf{h}_{t+1}^2)\\,\\nabla_{\\mathbf{h}_{t+1}}L$", fontsize=13, color=S.OUTC, va="center")
    ax.text(10.6, 3.9, "from the next state, back along $\\mathbf{U}$", fontsize=10.5, color=S.GREY, va="center")
    ax.text(10.6, 3.45, "the two add up to $\\nabla_{\\mathbf{h}_t}L$ (Eq. 10.20)", fontsize=10.5, color=S.OUTC, va="center")
    # the forward-pass note, bottom left
    ax.text(-2.85, 6.3, "forward (solid): left to right,", fontsize=11, color=D.ENC, va="center")
    ax.text(-2.85, 5.9, "each $\\mathbf{h}_t$ computed once and stored", fontsize=11, color=D.ENC, va="center")
    ax.text(-2.85, 5.3, "backward (dashed): right to left,", fontsize=11, color=S.OUTC, va="center")
    ax.text(-2.85, 4.9, "each $\\nabla_{\\mathbf{h}_t}L$ from the two terms", fontsize=11, color=S.OUTC, va="center")
    ax.text(-2.85, 4.3, "then the parameter gradients:", fontsize=11, color="black", va="center")
    ax.text(-2.85, 3.9, "sums over $t$ of per-step terms", fontsize=11, color="black", va="center")
    ax.set_xlim(-3.1, 15.0); ax.set_ylim(-1.8, 6.6); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
