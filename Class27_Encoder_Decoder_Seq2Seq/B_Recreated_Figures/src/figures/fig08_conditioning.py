"""Two ways to condition a recurrent network on a vector, in the layouts
of Goodfellow et al. (2016), Figs. 10.9 and 10.12.

`condition_input`   a fixed vector x fed as an extra input to every step
                    of the recurrence (with the output y^(t-1) fed back
                    as the sequence input), Fig. 10.9
`condition_encdec`  an encoder reads the input sequence into a context C,
                    which starts the decoder and enters every decoder
                    step, Fig. 10.12
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from common import style as S
from common import draw as D
from common import bookdraw as B

NAME = "condition_input"
NAMES = ("condition_input", "condition_encdec")


def node(ax, x, y, label, color, fill, r=0.32, fs=10.5, ls="-"):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.2, zorder=3, ls=ls))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def build_input():
    S.use()
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    xs = [0, 1.9, 3.8]; st = ["t-1", "t", "t+1"]
    node(ax, 1.9, -1.6, "$\\mathbf{x}$", B.CTX, B.CTX_F, r=0.36, fs=12)
    for x, s in zip(xs, st):
        node(ax, x, 0, "$\\mathbf{h}_{%s}$" % s, B.SRC, B.SRC_F)
        node(ax, x, 1.4, "$\\mathbf{o}_{%s}$" % s, B.TGT, B.TGT_F)
        node(ax, x, 2.6, "$L_{%s}$" % s, S.ACC, B.ATT_F, r=0.27)
        node(ax, x, 3.7, "$\\mathbf{y}_{%s}$" % s, S.GREY, "white", r=0.27)
        D.arrow(ax, (x, 0.34), (x, 1.06)); ax.text(x + 0.12, 0.7, "$\\mathbf{V}$", fontsize=9.5)
        D.arrow(ax, (x, 1.74), (x, 2.31)); D.arrow(ax, (x, 3.41), (x, 2.89))
        # the fixed vector into every step
        B.arrow(ax, (1.9 + 0.3 * (x - 1.9) / 1.9, -1.26), (x, -0.34), color=B.CTX, lw=1.4)
        ax.text((x + 1.9) / 2 + 0.18, -0.85, "$\\mathbf{R}$", fontsize=9.5, color=B.CTX)
    for p, q in zip(xs[:-1], xs[1:]):
        D.arrow(ax, (p + 0.34, 0), (q - 0.34, 0), color=B.SRC, lw=1.4); ax.text((p + q) / 2, 0.16, "$\\mathbf{U}$", ha="center", fontsize=9.5, color=B.SRC)
        # output feedback y_{t-1} -> h_t
        ax.annotate("", xy=(q - 0.22, 0.28), xytext=(p + 0.22, 3.55), arrowprops=dict(arrowstyle="-|>", color=S.GREY, lw=0.9, connectionstyle="arc3,rad=0.55", shrinkA=0, shrinkB=0), zorder=2)
        ax.text((p + q) / 2 + 0.75, 2.0, "$\\mathbf{W}$", fontsize=9.5, color=S.GREY)
    node(ax, -1.7, 0, "$\\mathbf{h}_{\\ldots}$", B.SRC, "white", ls=(0, (3, 2))); D.arrow(ax, (-1.36, 0), (-0.34, 0), color=B.SRC, lw=1.4)
    node(ax, 5.5, 0, "$\\mathbf{h}_{\\ldots}$", B.SRC, "white", ls=(0, (3, 2))); D.arrow(ax, (4.14, 0), (5.16, 0), color=B.SRC, lw=1.4)
    node(ax, 5.5, 3.7, "$\\mathbf{y}_{\\ldots}$", S.GREY, "white", r=0.27, ls=(0, (3, 2)))
    ax.annotate("", xy=(5.28, 0.28), xytext=(4.02, 3.55), arrowprops=dict(arrowstyle="-|>", color=S.GREY, lw=0.9, connectionstyle="arc3,rad=0.55", shrinkA=0, shrinkB=0), zorder=2)
    ax.text(7.0, 1.6, "one vector $\\mathbf{x}$, the same at every step,\nenters through $\\mathbf{R}$;\nthe previous output $\\mathbf{y}_{t-1}$ is the\nsequence input, through $\\mathbf{W}$",
            ha="left", va="center", fontsize=11, color=S.GREY)
    ax.set_xlim(-2.3, 11.5); ax.set_ylim(-2.2, 4.3); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "condition_input")


def build_encdec():
    S.use()
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    # encoder box
    ax.add_patch(FancyBboxPatch((-0.8, 2.6), 6.4, 2.6, boxstyle="round,pad=0.05,rounding_size=0.3", fc="none", ec=B.SRC, lw=1.3, zorder=1))
    ax.text(2.4, 5.45, "Encoder", ha="center", fontsize=12, color=B.SRC)
    xe = [0, 1.6, 3.2, 4.8]; le = ["1", "2", "\\ldots", "n_x"]
    for x, l in zip(xe, le):
        dashed = l == "\\ldots"
        node(ax, x, 4.4, "" if dashed else "", B.SRC, B.SRC_F if not dashed else "white", ls=(0, (3, 2)) if dashed else "-")
        if dashed: ax.text(x, 4.4, "$\\cdots$", ha="center", va="center", fontsize=12, color=B.SRC)
        node(ax, x, 3.2, "$\\mathbf{x}_{%s}$" % l, S.GREY, "#EEEEEE", ls=(0, (3, 2)) if dashed else "-")
        D.arrow(ax, (x, 3.54), (x, 4.06))
    for p, q in zip(xe[:-1], xe[1:]):
        D.arrow(ax, (p + 0.34, 4.4), (q - 0.34, 4.4), color=B.SRC, lw=1.3)
    # context
    node(ax, 7.4, 4.4, "$\\mathbf{C}$", B.CTX, B.CTX_F, r=0.4, fs=13)
    ax.plot([5.14, 6.6], [4.4, 4.4], color=B.SRC, lw=1.3); B.arrow(ax, (6.6, 4.4), (7.0, 4.4), color=B.SRC, lw=1.3)
    # decoder box
    ax.add_patch(FancyBboxPatch((-0.8, -1.4), 6.4, 3.0, boxstyle="round,pad=0.05,rounding_size=0.3", fc="none", ec=B.TGT, lw=1.3, zorder=1))
    ax.text(2.4, 1.85, "Decoder", ha="center", fontsize=12, color=B.TGT)
    xd = [0, 1.6, 3.2, 4.8]; ld = ["1", "2", "\\ldots", "n_y"]
    for x, l in zip(xd, ld):
        dashed = l == "\\ldots"
        node(ax, x, 0.6, "", B.TGT, B.TGT_F if not dashed else "white", ls=(0, (3, 2)) if dashed else "-")
        if dashed: ax.text(x, 0.6, "$\\cdots$", ha="center", va="center", fontsize=12, color=B.TGT)
        node(ax, x, -0.6, "$\\mathbf{y}_{%s}$" % l, S.GREY, "white", ls=(0, (3, 2)) if dashed else "-")
        D.arrow(ax, (x, 0.26), (x, -0.26), color=B.TGT)
        # C into every decoder step
        ax.annotate("", xy=(x + 0.1, 0.9), xytext=(7.1, 4.15), arrowprops=dict(arrowstyle="-|>", color=B.CTX, lw=0.9, ls=(0, (3, 2)), shrinkA=0, shrinkB=0), zorder=2)
    for p, q in zip(xd[:-1], xd[1:]):
        D.arrow(ax, (p + 0.34, 0.6), (q - 0.34, 0.6), color=B.TGT, lw=1.3)
        ax.annotate("", xy=(q - 0.2, 0.32), xytext=(p + 0.2, -0.86), arrowprops=dict(arrowstyle="-|>", color=S.GREY, lw=0.8, connectionstyle="arc3,rad=-0.3", shrinkA=0, shrinkB=0), zorder=2)
    # C as the decoder's initial state
    ax.plot([7.4, 7.4, -1.6, -1.6], [4.0, -1.9, -1.9, 0.6], color=B.CTX, lw=1.2, zorder=2); B.arrow(ax, (-1.6, 0.6), (-0.34, 0.6), color=B.CTX, lw=1.2)
    ax.text(7.7, 1.5, "solid: $\\mathbf{C}$ starts the decoder;\ndashed: $\\mathbf{C}$ enters every step;\nthe previous $\\mathbf{y}$ is the next input", ha="left", va="center", fontsize=11, color=S.GREY)
    ax.set_xlim(-2.0, 12.4); ax.set_ylim(-2.3, 5.9); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "condition_encdec")


def build():
    build_input(); build_encdec()


if __name__ == "__main__":
    build()
