"""attention_top — a different context c_i at every decoder step (Jurafsky Fig. 14.21 layout)

Used in: Class 27 (B_Recreated_Figures), slide(s): "Attention, drawn: a context per step"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python attention_top.py            -> output/attention_top.pdf
         python attention_top.py --out DIR  -> DIR/attention_top.pdf

Ported from Class27_Encoder_Decoder_Seq2Seq/B_Recreated_Figures/src/figures/fig12_attention_book.py (build_top()),
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
import common as S
import bookdraw as B

NAME = "attention_top"


def build():
    S.use()
    rng = np.random.default_rng(5)
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    xs = [0, 2.0, 4.0]; labs = ["1", "2", "i"]
    yc, yh, ys, yo = -1.1, 0.0, 1.15, 2.05
    B.band(ax, -0.7, 5.3, yh, h=1.0, color=B.TGT)
    for k, (x, l) in enumerate(zip(xs, labs)):
        B.hidden(ax, x, yh, color=B.TGT, label="$\\mathbf{h}^d_{%s}$" % l, w=0.56, fs=9.5)
        B.softmax(ax, x, ys, rng=rng); B.arrow(ax, (x, yh + 0.3), (x, ys - 0.2), lw=0.9)
        ax.text(x, yo, "$\\mathbf{y}_{%s}$" % l, ha="center", va="center", fontsize=11.5, color=B.TGT); B.arrow(ax, (x, ys + 0.2), (x, yo - 0.22), lw=0.9)
        ax.text(x, yc, "$\\mathbf{c}_{%s}$" % l, ha="center", va="center", fontsize=11.5, color=B.CTX)
        B.arrow(ax, (x, yc + 0.25), (x, yh - 0.3), color=B.CTX, lw=1.3)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.3, yh), (x - 0.3, yh), color=B.TGT, lw=1.3)
            B.arrow(ax, (xs[k - 1] + 0.4, yo - 0.15), (x - 0.42, yh + 0.32), color=S.GREY, lw=0.9, ls=(0, (4, 2)), z=2)
    ax.text(3.0, yh, "$\\cdots$", ha="center", va="center", fontsize=13, color=B.TGT, zorder=5)
    B.arrow(ax, (xs[-1] + 0.3, yh), (5.1, yh), color=B.TGT, lw=1.3); ax.text(5.4, yh, "$\\cdots$", ha="left", va="center", fontsize=13, color=B.TGT)
    ax.text(6.4, 0.5, "a different context at every step:\n$\\mathbf{c}_i$ is computed from all the\nencoder states, for step $i$", ha="left", va="center", fontsize=11, color=B.CTX)
    ax.set_xlim(-1.2, 11.0); ax.set_ylim(-1.6, 2.5); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "attention_top")


if __name__ == "__main__":
    build()
