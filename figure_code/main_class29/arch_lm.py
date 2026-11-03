"""arch_lm — language modelling, alone, on an example sentence

Used in: Class 29 (B_Recreated_Figures), slide(s): "(c) Language modelling"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python arch_lm.py            -> output/arch_lm.pdf
         python arch_lm.py --out DIR  -> DIR/arch_lm.pdf

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

NAME = "arch_lm"


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


def panel_lm(ax, xs, words, nexts, title):
    rnn(ax, xs[0] - 0.7, xs[-1] + 0.7, 0.0, B.SRC, "RNN")
    io(ax, xs, -1.2, words, "black", up=False)
    io(ax, xs, 1.2, nexts, B.TGT, up=True)
    ax.set_title(title, loc="left", fontsize=12)


def build():
    S.use()
    xs = [0, 1.6, 3.2, 4.8, 6.4]
    fig, ax = plt.subplots(figsize=(5.2, 1.85))
    panel_lm(ax, xs, ["So", "long", "and", "thanks", "for"], ["long", "and", "thanks", "for", "all"], "language modelling: the next token at every step")
    ax.set_xlim(-0.9, 7.3); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off"); S.save(fig, "arch_lm")


if __name__ == "__main__":
    build()
