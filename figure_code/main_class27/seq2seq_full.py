"""seq2seq_full — one RNN over source + separator + target (Jurafsky Fig. 14.17 layout)

Used in: Class 27 (B_Recreated_Figures), slide(s): "The whole model on the example"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python seq2seq_full.py            -> output/seq2seq_full.pdf
         python seq2seq_full.py --out DIR  -> DIR/seq2seq_full.pdf

Ported from Class27_Encoder_Decoder_Seq2Seq/B_Recreated_Figures/src/figures/fig09_seq2seq_book.py (build_full()),
same code, data, parameters and seeds.  The original module's description:

    The encoder-decoder for translation, in the layouts of Jurafsky &
    Martin (2026), Figs. 14.17 and 14.18.

    `seq2seq_full`      Fig. 14.17: source and target concatenated with a
                        separator, one recurrent network run over both; the
                        outputs of the source steps are ignored, the final
                        source state (teal) conditions the target steps; each
                        output word is the next input.
    `seq2seq_unrolled`  Fig. 14.18: encoder and decoder as two bands,
                        h^e_n = c = h^d_0, the context also entering every
                        decoder step; softmax and outputs above the decoder.
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
import bookdraw as B

NAME = "seq2seq_full"
SRCW = ["the", "green", "witch", "arrived"]
TGTW = ["llegó", "la", "bruja", "verde"]


def build():
    S.use()
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    dx = 1.45
    xs = [k * dx for k in range(4)]; xt = [4.6 * dx / 1.45 * 1.0 + k * dx for k in range(5)]
    xt = [xs[-1] + dx + k * dx for k in range(5)]
    yw, ye, yh, ys, yo = 0.0, 1.05, 2.3, 3.5, 4.4
    B.band(ax, xs[0] - 0.6, xt[-1] + 0.6, yh, h=1.0, color=B.SRC)
    # source steps
    for k, (x, w) in enumerate(zip(xs, SRCW)):
        last = k == 3
        B.column(ax, x, yw, ye, yh, word=w, color=B.SRC, hid_fill=B.CTX_F if last else None, hid_label="$\\mathbf{h}_n$" if last else None)
        if last:
            ax.add_patch(plt.Rectangle((x - 0.25, yh - 0.25), 0.5, 0.5, fc=B.CTX_F, ec=B.CTX, lw=1.5, zorder=3)); ax.text(x, yh, "$\\mathbf{h}_n$", ha="center", va="center", fontsize=8.5, color=B.CTX, zorder=4)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.SRC, lw=1.3)
    ax.text((xs[0] + xs[-1]) / 2, ys + 0.05, "(output of the source steps is ignored)", ha="center", fontsize=9.5, color=S.GREY)
    # target steps
    tin = ["$\\langle s\\rangle$"] + TGTW
    tout = TGTW + ["$\\langle/s\\rangle$"]
    for k, (x, wi, wo) in enumerate(zip(xt, tin, tout)):
        B.column(ax, x, yw, ye, yh, ys, yo, word=wi, out=wo, color=B.TGT, rng=rng)
        prev = xs[-1] if k == 0 else xt[k - 1]
        B.arrow(ax, (prev + 0.27, yh), (x - 0.27, yh), color=B.SRC if k == 0 else B.TGT, lw=1.3)
        if k < 4:
            B.arrow(ax, (x + 0.42, yo - 0.15), (xt[k + 1] - 0.42, yw + 0.15), color=B.TGT, lw=1.0, ls=(0, (4, 2)), z=2)
    B.brace(ax, xs[0] - 0.4, xs[-1] + 0.4, yw - 0.45, "source text", color=B.SRC)
    B.brace(ax, xt[1] - 0.4, xt[-1] + 0.4, yo + 0.4, "target text", color=B.TGT, up=True)
    ax.text(xt[0], yw - 0.75, "separator", ha="center", fontsize=9.5, color=S.GREY)
    for y, lab in ((ye, "embedding\nlayer"), (yh, "hidden\nlayer(s)"), (ys, "softmax")):
        ax.text(xs[0] - 0.95, y, lab, ha="right", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(xs[0] - 2.6, xt[-1] + 0.9); ax.set_ylim(-1.5, 5.6); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "seq2seq_full")


if __name__ == "__main__":
    build()
