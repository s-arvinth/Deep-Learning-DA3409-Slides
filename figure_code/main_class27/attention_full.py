"""attention_full — one attention decoder step in full (Jurafsky Fig. 14.22 layout)

Used in: Class 27 (B_Recreated_Figures), slide(s): "Attention, drawn: one decoder step in full"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python attention_full.py            -> output/attention_full.pdf
         python attention_full.py --out DIR  -> DIR/attention_full.pdf

Ported from Class27_Encoder_Decoder_Seq2Seq/B_Recreated_Figures/src/figures/fig12_attention_book.py (build_full()),
same code, data, parameters and seeds.  The original module's description:

    Attention, in the layouts of Jurafsky & Martin (2026), Figs. 14.21
    and 14.22.

    `attention_top`   Fig. 14.21: the decoder with a different context c_i
                      at every step.
    `attention_full`  Fig. 14.22: one decoder step in full -- the encoder
                      states, a score of each against the previous decoder
                      state, the softmax weights alpha_ij, and the weighted
                      sum that is c_i, which enters the decoder step.  The
                      weights are written as symbols, not numbers.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import common as S
import bookdraw as B

NAME = "attention_full"


def build():
    S.use()
    rng = np.random.default_rng(6)
    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    xe = [0, 1.6, 3.2, 5.2]; le = ["1", "2", "3", "n"]
    xd = [8.6, 10.6]
    yx, yh, ya, yc = -1.0, 0.3, 2.2, 4.0
    B.band(ax, -0.6, 5.8, yh, h=1.0, color=B.SRC)
    for k, (x, l) in enumerate(zip(xe, le)):
        B.hidden(ax, x, yh, color=B.SRC, label="$\\mathbf{h}^e_{%s}$" % l, w=0.56, fs=9.5)
        ax.text(x, yx, "$\\mathbf{x}_{%s}$" % l, ha="center", va="center", fontsize=11.5, color=B.SRC); B.arrow(ax, (x, yx + 0.25), (x, yh - 0.3), lw=0.9)
        if k:
            B.arrow(ax, (xe[k - 1] + 0.3, yh), (x - 0.3, yh), color=B.SRC, lw=1.3)
        # attention weight circle above each state
        ax.add_patch(Circle((x, ya), 0.34, fc=B.ATT_F, ec=B.ATT, lw=1.4, ls=(0, (3, 2)), zorder=3))
        ax.text(x, ya, "$\\alpha_{i%s}$" % l, ha="center", va="center", fontsize=10, color=B.ATT, zorder=4)
        # score: previous decoder state dotted with this encoder state
        B.arrow(ax, (xd[0] - 0.3, 0.55), (x + 0.15, ya - 0.34), color=B.ATT, lw=0.8, ls=(0, (2, 2)), z=2)
        B.arrow(ax, (x, yh + 0.3), (x, ya - 0.36), color=B.SRC, lw=0.8)
        # weighted sum into c_i
        B.arrow(ax, (x + 0.1, ya + 0.34), (6.6, yc - 0.15), color=B.ATT, lw=1.0)
    ax.text(4.2, yh, "$\\cdots$", ha="center", va="center", fontsize=13, color=B.SRC, zorder=5); ax.text(4.2, yx, "$\\cdots$", ha="center", va="center", fontsize=13, color=B.SRC)
    ax.text(-1.0, ya, "attention\nweights $\\alpha_{ij}$", ha="right", va="center", fontsize=10.5, color=B.ATT)
    ax.text(-1.0, yh, "hidden\nlayer(s)", ha="right", va="center", fontsize=10.5, color=S.GREY)
    ax.text(1.5, ya - 0.95, "$\\mathrm{score}(\\mathbf{h}^d_{i-1}, \\mathbf{h}^e_j) = \\mathbf{h}^d_{i-1}\\cdot\\mathbf{h}^e_j$", ha="center", fontsize=10.5, color=B.ATT)
    # c_i
    ax.add_patch(FancyBboxPatch((6.6, yc - 0.32), 0.9, 0.64, boxstyle="round,pad=0.03,rounding_size=0.15", fc=B.CTX_F, ec=B.CTX, lw=1.5, zorder=3))
    ax.text(7.05, yc, "$\\mathbf{c}_i$", ha="center", va="center", fontsize=12, color=B.CTX, zorder=4)
    ax.text(4.4, yc + 0.35, "$\\mathbf{c}_i = \\sum_j \\alpha_{ij}\\,\\mathbf{h}^e_j$", ha="center", va="bottom", fontsize=12, color=B.ATT)
    # decoder
    B.band(ax, xd[0] - 0.7, xd[-1] + 0.7, yh, h=1.0, color=B.TGT)
    for k, (x, l) in enumerate(zip(xd, ["i-1", "i"])):
        B.hidden(ax, x, yh, color=B.TGT, label="$\\mathbf{h}^d_{%s}$" % l, w=0.56, fs=9.5)
        B.softmax(ax, x, 1.55, rng=rng); B.arrow(ax, (x, yh + 0.3), (x, 1.35), lw=0.9)
        ax.text(x, 2.45, "$\\mathbf{y}_{%s}$" % l, ha="center", va="center", fontsize=11.5, color=B.TGT); B.arrow(ax, (x, 1.75), (x, 2.23), lw=0.9)
        ax.text(x + 0.5, yx, "$\\mathbf{y}_{%s}$" % ("i-2" if k == 0 else "i-1"), ha="center", va="center", fontsize=11.5, color=B.TGT)
        B.arrow(ax, (x + 0.18, yx + 0.25), (x + 0.18, yh - 0.3), lw=0.9)
    B.arrow(ax, (xd[0] + 0.3, yh), (xd[1] - 0.3, yh), color=B.TGT, lw=1.3)
    B.arrow(ax, (xd[0] + 0.4, 2.3), (xd[1] + 0.2, yx + 0.25), color=S.GREY, lw=0.9, ls=(0, (4, 2)), z=2)
    ax.text(7.6, yh, "$\\cdots$", ha="center", va="center", fontsize=13, color=B.TGT)
    # c_i into decoder step i: right from the box, down beside the band, along under it, up into the box
    ax.plot([7.5, 7.9, 7.9, xd[1] - 0.18], [yc, yc, yh - 0.72, yh - 0.72], color=B.CTX, lw=1.3, zorder=2)
    B.arrow(ax, (xd[1] - 0.18, yh - 0.72), (xd[1] - 0.18, yh - 0.31), color=B.CTX, lw=1.3)
    ax.text(xd[1] - 0.45, yh - 0.62, "$\\mathbf{c}_i$", ha="right", va="center", fontsize=11, color=B.CTX)
    # c_{i-1} into step i-1, from the left
    B.arrow(ax, (xd[0] - 0.18, yx + 0.25), (xd[0] - 0.18, yh - 0.31), color=B.CTX, lw=1.0)
    ax.text(xd[0] - 0.62, yx, "$\\mathbf{c}_{i-1}$", ha="center", va="center", fontsize=10, color=B.CTX)
    B.brace(ax, -0.5, 5.7, yx - 0.4, "encoder", color=B.SRC)
    B.brace(ax, xd[0] - 0.6, xd[-1] + 0.6, 2.9, "decoder", color=B.TGT, up=True)
    ax.set_xlim(-3.2, 11.8); ax.set_ylim(-2.4, 5.2); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "attention_full")


if __name__ == "__main__":
    build()
