"""The four RNN architectures for NLP tasks, in the layout of Jurafsky
& Martin (2026), Fig. 14.15: sequence labelling, sequence
classification, language modelling and the encoder-decoder."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from common import style as S
from common import draw as D

NAME = "four_architectures"


def chain(ax, x0, n, color, fill, y=0.0, dx=0.9, label="RNN"):
    D.band(ax, x0 - 0.45, x0 + (n - 1) * dx + 0.45, y - 0.32, y + 0.32, color, alpha=0.13)
    cs = []
    for k in range(n):
        x = x0 + k * dx
        ax.add_patch(plt.Rectangle((x - 0.18, y - 0.2), 0.36, 0.4, fc=fill, ec=color, lw=1.1, zorder=3))
        if k:
            D.arrow(ax, (x - dx + 0.18, y), (x - 0.18, y), color=color)
        cs.append((x, y))
    ax.text(x0 + (n - 1) * dx / 2, y + 0.42, label, ha="center", va="bottom", fontsize=10.5, color=color)
    return cs


def tokens(ax, cs, labs, dy, color="black"):
    for (x, y), lab in zip(cs, labs):
        ax.text(x, y + dy, lab, ha="center", va="center", fontsize=11, color=color)
        if dy < 0:
            D.arrow(ax, (x, y + dy + 0.18), (x, y - 0.22))
        else:
            D.arrow(ax, (x, y + 0.22), (x, y + dy - 0.18), color=color)


def build():
    S.use()
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 5.6))
    X = ["$x_1$", "$x_2$", "$\\cdots$", "$x_n$"]
    # (a) sequence labelling
    ax = axs[0, 0]; cs = chain(ax, 0, 4, D.ENC, D.ENC_F)
    tokens(ax, cs, X, -0.75); tokens(ax, cs, ["$y_1$", "$y_2$", "$\\cdots$", "$y_n$"], 0.95, D.DEC)
    ax.set_title("(a) sequence labelling", loc="left", fontsize=13)
    # (b) sequence classification
    ax = axs[0, 1]; cs = chain(ax, 0, 4, D.ENC, D.ENC_F)
    tokens(ax, cs, X, -0.75)
    ax.text(cs[-1][0], 0.95, "$y$", ha="center", va="center", fontsize=11, color=D.DEC)
    D.arrow(ax, (cs[-1][0], 0.22), (cs[-1][0], 0.77), color=D.DEC)
    ax.text(cs[-1][0] + 0.25, 0.5, "$\\mathbf{h}_n$", fontsize=10, color=D.DEC)
    ax.set_title("(b) sequence classification", loc="left", fontsize=13)
    # (c) language modelling
    ax = axs[1, 0]; cs = chain(ax, 0, 4, D.ENC, D.ENC_F)
    tokens(ax, cs, ["$x_1$", "$x_2$", "$\\cdots$", "$x_{t-1}$"], -0.75)
    tokens(ax, cs, ["$x_2$", "$x_3$", "$\\cdots$", "$x_t$"], 0.95, D.DEC)
    ax.set_title("(c) language modelling", loc="left", fontsize=13)
    # (d) encoder-decoder
    ax = axs[1, 1]; ce = chain(ax, 0, 3, D.ENC, D.ENC_F, y=-0.45, dx=0.8, label="encoder RNN")
    tokens(ax, ce, ["$x_1$", "$x_2$", "$x_n$"], -0.6)
    cd = chain(ax, 3.0, 3, D.DEC, D.DEC_F, y=0.6, dx=0.8, label="decoder RNN")
    tokens(ax, cd, ["$y_1$", "$y_2$", "$y_m$"], 0.95, D.DEC)
    ax.add_patch(plt.Rectangle((2.0, -0.08), 0.62, 0.3, fc="white", ec=S.GREY, lw=1.0, zorder=3))
    ax.text(2.31, 0.07, "context", ha="center", va="center", fontsize=8.5, color=S.GREY)
    D.arrow(ax, (ce[-1][0] + 0.18, -0.45), (2.0, 0.0), color=D.ENC)
    D.arrow(ax, (2.62, 0.12), (cd[0][0] - 0.18, 0.55), color=D.DEC)
    ax.set_title("(d) encoder–decoder", loc="left", fontsize=13)
    for ax in axs.ravel():
        ax.set_xlim(-0.8, 5.6); ax.set_ylim(-1.15, 1.6); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(h_pad=0.4, w_pad=0.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
