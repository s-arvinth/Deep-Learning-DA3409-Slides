"""condition_input — a fixed vector fed to every step of an RNN (Goodfellow Fig. 10.9 layout)

Used in: Class 27 (B_Recreated_Figures), slide(s): "Two ways to hand over a vector: as an extra input"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python condition_input.py            -> output/condition_input.pdf
         python condition_input.py --out DIR  -> DIR/condition_input.pdf

Ported from Class27_Encoder_Decoder_Seq2Seq/B_Recreated_Figures/src/figures/fig08_conditioning.py (build_input()),
same code, data, parameters and seeds.  The original module's description:

    Two ways to condition a recurrent network on a vector, in the layouts
    of Goodfellow et al. (2016), Figs. 10.9 and 10.12.

    `condition_input`   a fixed vector x fed as an extra input to every step
                        of the recurrence (with the output y^(t-1) fed back
                        as the sequence input), Fig. 10.9
    `condition_encdec`  an encoder reads the input sequence into a context C,
                        which starts the decoder and enters every decoder
                        step, Fig. 10.12
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import common as S
import draw as D
import bookdraw as B

NAME = "condition_input"


def node(ax, x, y, label, color, fill, r=0.32, fs=10.5, ls="-"):
    ax.add_patch(Circle((x, y), r, fc=fill, ec=color, lw=1.2, zorder=3, ls=ls))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def build():
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


if __name__ == "__main__":
    build()
