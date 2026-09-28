"""Three ways to give the gradient a path that survives, drawn:
(a) skip connections across time, an edge from t-d to t, so a fact
travels in tau/d hops instead of tau; (b) a leaky unit, a linear
self-loop of fixed weight alpha near one, along which the derivative is
alpha; (c) a gate, the same self-loop with a weight the network computes
from the input at every step."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from common import style as S
from common import draw as D

NAME = "gradient_paths"


def node(ax, x, y, label, color, fill, r=0.3, fs=10.5):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.2, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def build():
    S.use()
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.2))
    # (a) skip connections
    ax = axs[0]; xs = [k * 1.1 for k in range(7)]
    for k, x in enumerate(xs):
        node(ax, x, 0, "$\\mathbf{h}_{%d}$" % (k + 1), D.ENC, D.ENC_F, r=0.32, fs=9)
        if k:
            D.arrow(ax, (xs[k - 1] + 0.33, 0), (x - 0.33, 0), color=D.ENC)
    for k in range(0, 4, 3):
        ax.annotate("", xy=(xs[k + 3], 0.36), xytext=(xs[k], 0.36),
                    arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.8, connectionstyle="arc3,rad=-0.45", shrinkA=0, shrinkB=0), zorder=5)
    ax.text(3.3, 1.75, "an edge from $t - d$ to $t$ ($d = 3$)", ha="center", fontsize=11, color=S.OUTC)
    ax.text(3.3, -0.85, "$\\tau$ steps along the chain, $\\tau / d$ along the skips:\nthe product has $\\tau/d$ factors instead of $\\tau$", ha="center", va="top", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-0.6, 7.2); ax.set_ylim(-1.9, 2.2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) skip connections across time", loc="left", fontsize=12)
    # (b) leaky unit
    ax = axs[1]
    node(ax, 0, 0, "$\\mu_t$", D.ENC, D.ENC_F, r=0.42, fs=12)
    ax.annotate("", xy=(0.12, 0.44), xytext=(-0.12, 0.44), arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=2.0, connectionstyle="arc3,rad=-2.2", shrinkA=0, shrinkB=0), zorder=5)
    ax.text(0, 1.45, "$\\alpha$", ha="center", fontsize=13, color=S.OUTC)
    D.arrow(ax, (-1.6, 0), (-0.45, 0)); ax.text(-1.05, 0.22, "$(1-\\alpha)\\,v_t$", ha="center", fontsize=10.5)
    ax.text(0, -0.95, "$\\mu_t = \\alpha\\,\\mu_{t-1} + (1 - \\alpha)\\,v_t$", ha="center", fontsize=11.5)
    ax.text(0, -1.45, "$\\partial\\mu_t/\\partial\\mu_{t-1} = \\alpha$, a constant near one", ha="center", fontsize=10.5, color=S.GREY)
    # inset: impulse responses
    ins = ax.inset_axes([0.58, 0.42, 0.42, 0.52])
    tt = np.arange(40)
    for al, c in ((0.5, S.GREY), (0.9, D.ENC), (0.98, S.OUTC)):
        ins.plot(tt, al ** tt, color=c, lw=1.6, label="$\\alpha = %.2f$" % al)
    ins.set_xlabel("steps since the input", fontsize=8); ins.set_ylabel("what remains", fontsize=8)
    ins.tick_params(labelsize=7); ins.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_xlim(-2.0, 3.6); ax.set_ylim(-1.9, 2.2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) a leaky unit: a fixed self-loop", loc="left", fontsize=12)
    # (c) a gate
    ax = axs[2]
    node(ax, 0, 0, "$\\mathbf{c}_t$", D.ENC, D.ENC_F, r=0.42, fs=12)
    ax.annotate("", xy=(0.12, 0.44), xytext=(-0.12, 0.44), arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=2.0, connectionstyle="arc3,rad=-2.2", shrinkA=0, shrinkB=0), zorder=5)
    ax.add_patch(FancyBboxPatch((-0.55, 1.2), 1.1, 0.5, boxstyle="round,pad=0.03,rounding_size=0.12", fc="#F1D6D6", ec=S.OUTC, lw=1.2, zorder=6))
    ax.text(0, 1.45, "$\\mathbf{f}_t = \\sigma(\\cdot)$", ha="center", va="center", fontsize=10.5, color=S.OUTC, zorder=7)
    D.arrow(ax, (1.9, 1.45), (0.6, 1.45), color=S.OUTC); ax.text(2.0, 1.45, "$\\mathbf{x}_t,\\ \\mathbf{h}_{t-1}$", ha="left", va="center", fontsize=10.5, color=S.OUTC)
    D.arrow(ax, (-1.6, 0), (-0.45, 0)); ax.text(-1.05, 0.22, "$\\mathbf{i}_t \\odot \\mathbf{g}_t$", ha="center", fontsize=10.5)
    ax.text(0, -0.95, "$\\mathbf{c}_t = \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \\mathbf{g}_t$", ha="center", fontsize=11.5)
    ax.text(0, -1.45, "$\\partial\\mathbf{c}_t/\\partial\\mathbf{c}_{t-1} = \\mathrm{diag}(\\mathbf{f}_t)$, chosen at every step", ha="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-2.0, 3.6); ax.set_ylim(-1.9, 2.2); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(c) a gate: the self-loop's weight is decided by the input", loc="left", fontsize=12)
    fig.tight_layout(w_pad=0.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
