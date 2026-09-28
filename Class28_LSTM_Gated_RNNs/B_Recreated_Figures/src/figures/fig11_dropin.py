"""Basic neural units, in the layout of Jurafsky & Martin (2026), Fig.
14.14: (a) a feedforward unit, a weighted sum and an activation; (b) a
simple recurrent unit, the same with two inputs; (c) an LSTM unit,
whose internals are hidden: two vectors in and two out."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from common import style as S

NAME = "dropin_modules"


def arrow(ax, p, q, color="black", lw=1.4):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1), zorder=5)


def big(ax, x, y, color, fill, r=1.15):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.6, zorder=2))


def small(ax, x, y, label, r=0.36):
    ax.add_patch(Circle((x, y), r, fc="white", ec="black", lw=1.1, zorder=3)); ax.text(x, y, label, ha="center", va="center", fontsize=12, zorder=4)


def build():
    S.use()
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.4))
    # (a) feedforward
    ax = axs[0]; big(ax, 0, 0, S.L1, "#E4DDEB")
    small(ax, 0, -0.5, "$\\Sigma$"); small(ax, 0, 0.5, "$g$")
    arrow(ax, (0, -0.14), (0, 0.14)); ax.text(-0.2, 0.0, "$z$", ha="right", va="center", fontsize=11)
    arrow(ax, (0, -2.1), (0, -0.88)); ax.text(0, -2.35, "$\\mathbf{x}$", ha="center", va="top", fontsize=12)
    arrow(ax, (0, 0.88), (0, 2.1)); ax.text(-0.2, 1.4, "$a$", ha="right", va="center", fontsize=11); ax.text(0, 2.35, "$\\mathbf{h}$", ha="center", va="bottom", fontsize=12)
    ax.set_title("(a) a feedforward unit", loc="left", fontsize=12)
    # (b) simple recurrent
    ax = axs[1]; big(ax, 0, 0, S.L1, "#E4DDEB")
    small(ax, 0, -0.5, "$\\Sigma$"); small(ax, 0, 0.5, "$g$")
    arrow(ax, (0, -0.14), (0, 0.14)); ax.text(-0.2, 0.0, "$z$", ha="right", va="center", fontsize=11)
    arrow(ax, (-1.0, -2.1), (-0.2, -0.85)); arrow(ax, (1.0, -2.1), (0.2, -0.85))
    ax.text(-1.1, -2.35, "$\\mathbf{h}_{t-1}$", ha="center", va="top", fontsize=12); ax.text(1.1, -2.35, "$\\mathbf{x}_t$", ha="center", va="top", fontsize=12)
    arrow(ax, (0, 0.88), (0, 2.1)); ax.text(-0.2, 1.4, "$a$", ha="right", va="center", fontsize=11); ax.text(0, 2.35, "$\\mathbf{h}_t$", ha="center", va="bottom", fontsize=12)
    ax.set_title("(b) a simple recurrent unit", loc="left", fontsize=12)
    # (c) LSTM
    ax = axs[2]; big(ax, 0, 0, S.OUTC, "#F1D6D6", r=1.25)
    ax.text(0, 0, "LSTM\nunit", ha="center", va="center", fontsize=13, color=S.OUTC, zorder=4)
    for x0, lab in ((-1.3, "$\\mathbf{c}_{t-1}$"), (0.0, "$\\mathbf{h}_{t-1}$"), (1.3, "$\\mathbf{x}_t$")):
        arrow(ax, (x0, -2.1), (x0 * 0.45, -1.0 if x0 else -1.25)); ax.text(x0, -2.35, lab, ha="center", va="top", fontsize=12)
    for x0, lab in ((-0.9, "$\\mathbf{c}_t$"), (0.9, "$\\mathbf{h}_t$")):
        arrow(ax, (x0 * 0.5, 1.05 if x0 else 1.25), (x0, 2.1)); ax.text(x0, 2.35, lab, ha="center", va="bottom", fontsize=12)
    ax.set_title("(c) an LSTM unit", loc="left", fontsize=12)
    for ax in axs:
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.9, 2.9); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(w_pad=0.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
