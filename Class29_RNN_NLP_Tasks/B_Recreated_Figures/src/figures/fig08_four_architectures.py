"""The four RNN architectures for NLP tasks, in the layout of Jurafsky &
Martin (2026), Fig. 14.15, as one four-panel summary and as four
expanded, example-driven figures.

`four_architectures`  the summary: (a) sequence labelling, (b) sequence
                      classification, (c) language modelling, (d) the
                      encoder-decoder
`arch_labelling`, `arch_classification`, `arch_lm`, `arch_encdec`
                      each one on its own, with an example sentence
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from common import style as S
from common import bookdraw as B

NAME = "four_architectures"
NAMES = ("four_architectures", "arch_labelling", "arch_classification", "arch_lm", "arch_encdec")


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


def trap(ax, x, y, color, w=0.8, h=0.5):
    ax.add_patch(Polygon([(x - w / 2, y + h / 2), (x + w / 2, y + h / 2), (x + w / 4, y - h / 2), (x - w / 4, y - h / 2)], closed=True, fc=B.TGT_F, ec=color, lw=1.1, zorder=3))


def panel_labelling(ax, xs, words, tags, title):
    rnn(ax, xs[0] - 0.7, xs[-1] + 0.7, 0.0, B.SRC, "RNN")
    io(ax, xs, -1.2, words, "black", up=False)
    io(ax, xs, 1.2, tags, B.TGT, up=True)
    ax.set_title(title, loc="left", fontsize=12)


def panel_classification(ax, xs, words, label, title):
    rnn(ax, xs[0] - 0.7, xs[-1] + 0.7, 0.0, B.SRC, "RNN")
    io(ax, xs, -1.2, words, "black", up=False)
    trap(ax, xs[-1], 1.0, B.TGT); B.arrow(ax, (xs[-1], 0.4), (xs[-1], 0.73), color=B.TGT, lw=1.0)
    B.arrow(ax, (xs[-1], 1.27), (xs[-1], 1.6), color=B.TGT, lw=1.0)
    ax.text(xs[-1], 1.85, label, ha="center", va="center", fontsize=11, color=B.TGT)
    ax.set_title(title, loc="left", fontsize=12)


def panel_lm(ax, xs, words, nexts, title):
    rnn(ax, xs[0] - 0.7, xs[-1] + 0.7, 0.0, B.SRC, "RNN")
    io(ax, xs, -1.2, words, "black", up=False)
    io(ax, xs, 1.2, nexts, B.TGT, up=True)
    ax.set_title(title, loc="left", fontsize=12)


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


def build_summary():
    S.use()
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 6.2))
    xs = [0, 1.2, 2.6, 4.0]
    panel_labelling(axs[0, 0], xs, ["$\\mathbf{x}_1$", "$\\mathbf{x}_2$", "$\\cdots$", "$\\mathbf{x}_n$"], ["$\\mathbf{y}_1$", "$\\mathbf{y}_2$", "$\\cdots$", "$\\mathbf{y}_n$"], "(a) sequence labelling")
    panel_classification(axs[0, 1], xs, ["$\\mathbf{x}_1$", "$\\mathbf{x}_2$", "$\\cdots$", "$\\mathbf{x}_n$"], "$\\mathbf{y}$", "(b) sequence classification")
    panel_lm(axs[1, 0], xs, ["$\\mathbf{x}_1$", "$\\mathbf{x}_2$", "$\\cdots$", "$\\mathbf{x}_{t-1}$"], ["$\\mathbf{x}_2$", "$\\mathbf{x}_3$", "$\\cdots$", "$\\mathbf{x}_t$"], "(c) language modelling")
    panel_encdec(axs[1, 1], [0, 1.0, 2.0, 3.0], ["$\\mathbf{x}_1$", "$\\mathbf{x}_2$", "$\\cdots$", "$\\mathbf{x}_n$"], [4.6, 5.6, 6.6, 7.6], ["$\\mathbf{y}_1$", "$\\mathbf{y}_2$", "$\\cdots$", "$\\mathbf{y}_m$"], "(d) encoder--decoder")
    for ax in axs.ravel():
        ax.set_xlim(-1.0, 8.6); ax.set_ylim(-2.0, 2.9); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    S.save(fig, "four_architectures")


def build_expanded():
    S.use()
    xs = [0, 1.6, 3.2, 4.8, 6.4]
    fig, ax = plt.subplots(figsize=(5.2, 1.85))
    panel_labelling(ax, xs, ["Janet", "will", "back", "the", "bill"], ["NNP", "MD", "VB", "DT", "NN"], "sequence labelling: one output per input, aligned")
    ax.set_xlim(-0.9, 7.3); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off"); S.save(fig, "arch_labelling")
    fig, ax = plt.subplots(figsize=(5.2, 1.85))
    panel_classification(ax, xs, ["a", "wonderfully", "warm", "little", "film"], "positive", "sequence classification: one output for the whole input")
    ax.set_xlim(-0.9, 7.3); ax.set_ylim(-1.5, 2.2); ax.set_aspect("equal"); ax.axis("off"); S.save(fig, "arch_classification")
    fig, ax = plt.subplots(figsize=(5.2, 1.85))
    panel_lm(ax, xs, ["So", "long", "and", "thanks", "for"], ["long", "and", "thanks", "for", "all"], "language modelling: the next token at every step")
    ax.set_xlim(-0.9, 7.3); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off"); S.save(fig, "arch_lm")
    fig, ax = plt.subplots(figsize=(5.2, 2.25))
    panel_encdec(ax, [0, 1.4, 2.8, 4.2], ["the", "green", "witch", "arrived"], [6.2, 7.6, 9.0, 10.4], ["lleg\u00f3", "la", "bruja", "verde"], "encoder--decoder: a new sequence, of its own length and order")
    ax.set_xlim(-0.9, 11.2); ax.set_ylim(-1.8, 2.7); ax.set_aspect("equal"); ax.axis("off"); S.save(fig, "arch_encdec")


def build():
    build_summary(); build_expanded()


if __name__ == "__main__":
    build()
