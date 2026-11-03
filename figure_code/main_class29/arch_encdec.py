"""arch_encdec — the encoder-decoder, alone, on an example sentence

Used in: Class 29 (B_Recreated_Figures), slide(s): "(d) Encoder--decoder"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python arch_encdec.py            -> output/arch_encdec.pdf
         python arch_encdec.py --out DIR  -> DIR/arch_encdec.pdf

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/figures/fig08_four_architectures.py (build_expanded()),
same code, data, parameters and seeds.  The original module's description:

    The four RNN architectures for NLP tasks, in the layout of Jurafsky &
    Martin (2026), Fig. 14.15, as one four-panel summary and as four
    expanded, example-driven figures.

    `four_architectures`  the summary: (a) sequence labelling, (b) sequence
                          classification, (c) language modelling, (d) the
                          encoder-decoder
    `arch_labelling`, `arch_classification`, `arch_lm`, `arch_encdec`
                          each one on its own, with an example sentence
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import common as S
import bookdraw as B

NAME = "arch_encdec"


def rnn(ax, x0, x1, y, color, label, h=0.8):
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), x1 - x0, h, boxstyle="round,pad=0.05,rounding_size=0.4", fc=color, ec=color, lw=1.2, alpha=0.18, zorder=1))
    ax.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=11.5, color=color, zorder=2)


def io(ax, xs, y, labels, color, up=True, band_y=0.0, h=0.8, fs=11):
    for x, l in zip(xs, labels):
        ax.text(x, y, l, ha="center", va="center", fontsize=fs, color=color)
        if up:
            B.arrow(ax, (x, band_y + h / 2), (x, y - 0.3), color=color, lw=1.0)
        else:
            B.arrow(ax, (x, y + 0.3), (x, band_y - h / 2), color="black", lw=1.0)


def panel_encdec(ax, xs, words, ys, outs, title):
    rnn(ax, xs[0] - 0.7, xs[-1] + 0.7, -0.4, B.SRC, "encoder RNN", h=0.7)
    io(ax, xs, -1.5, words, "black", up=False, band_y=-0.4, h=0.7)
    rnn(ax, ys[0] - 0.7, ys[-1] + 0.7, 1.3, B.TGT, "decoder RNN", h=0.7)
    io(ax, ys, 2.4, outs, B.TGT, up=True, band_y=1.3, h=0.7)
    cx = (xs[-1] + ys[0]) / 2 + 0.2
    ax.add_patch(FancyBboxPatch((cx - 0.55, 0.25), 1.1, 0.4, boxstyle="round,pad=0.03,rounding_size=0.15", fc="white", ec=B.CTX, lw=1.1, zorder=3))
    ax.text(cx, 0.45, "context", ha="center", va="center", fontsize=9.5, color=B.CTX, zorder=4)
    B.arrow(ax, (xs[-1] + 0.5, -0.05), (cx - 0.3, 0.25), color=B.SRC, lw=1.1)
    B.arrow(ax, (cx + 0.3, 0.65), (ys[0] - 0.5, 0.95), color=B.CTX, lw=1.1)
    ax.set_title(title, loc="left", fontsize=12)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(5.2, 2.25))
    panel_encdec(ax, [0, 1.4, 2.8, 4.2], ["the", "green", "witch", "arrived"], [6.2, 7.6, 9.0, 10.4], ["lleg\u00f3", "la", "bruja", "verde"], "encoder--decoder: a new sequence, of its own length and order")
    ax.set_xlim(-0.9, 11.2); ax.set_ylim(-1.8, 2.7); ax.set_aspect("equal"); ax.axis("off"); S.save(fig, "arch_encdec")


if __name__ == "__main__":
    build()
