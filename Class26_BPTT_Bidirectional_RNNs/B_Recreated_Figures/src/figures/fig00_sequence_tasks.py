"""What a sequence model is asked to do: four shapes of problem, each an
input sequence mapped to something -- a label per token, one label for
the whole sequence, the next token, or a new sequence."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import style as S
from common import draw as D

NAME = "sequence_tasks"


def tok(ax, x, y, text, color, fill, w=1.05, h=0.5, fs=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fill, ec=color, lw=1.1, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def build():
    S.use()
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 5.6))
    X = {"grey": (S.GREY, "#EEEEEE"), "out": (D.DEC, D.DEC_F)}
    # (a) tagging
    ax = axs[0, 0]; words = ["Janet", "will", "back", "the", "bill"]; tags = ["NNP", "MD", "VB", "DT", "NN"]
    for k, (w, t) in enumerate(zip(words, tags)):
        tok(ax, k * 1.25, 0, w, *X["grey"]); tok(ax, k * 1.25, 1.4, t, *X["out"]); D.arrow(ax, (k * 1.25, 0.28), (k * 1.25, 1.12), color=D.DEC)
    ax.set_title("(a) a label for every token: part-of-speech tagging", loc="left", fontsize=12)
    # (b) classification
    ax = axs[0, 1]; words = ["a", "wonderfully", "warm", "film"]
    for k, w in enumerate(words):
        tok(ax, k * 1.25, 0, w, *X["grey"])
        ax.annotate("", xy=(3.75 + 1.2, 1.15), xytext=(k * 1.25, 0.28), arrowprops=dict(arrowstyle="-|>", color=D.DEC, lw=0.9, shrinkA=0, shrinkB=2))
    tok(ax, 4.95, 1.4, "positive", *X["out"], w=1.4)
    ax.set_title("(b) one label for the whole sequence: sentiment", loc="left", fontsize=12)
    # (c) next token
    ax = axs[1, 0]; words = ["Thanks", "for", "all", "the"]
    prev = None
    for k, w in enumerate(words):
        tok(ax, k * 1.25, 0, w, *X["grey"])
    tok(ax, 5.0, 0, "?", *X["out"], w=0.8)
    ax.annotate("", xy=(4.6, 0.0), xytext=(4.1, 0.0), arrowprops=dict(arrowstyle="-|>", color=D.DEC, lw=1.2))
    ax.text(5.0, 1.15, "fish  0.31\nmemories  0.12\nhelp  0.09", ha="center", va="center", fontsize=9.5, color=D.DEC)
    ax.text(1.9, 1.3, "a distribution over the next word,\nfrom everything before it", ha="center", fontsize=10.5, color=S.GREY)
    ax.set_title("(c) the next token: language modelling", loc="left", fontsize=12)
    # (d) sequence to sequence
    ax = axs[1, 1]; src = ["the", "green", "witch", "arrived"]; tgt = ["llegó", "la", "bruja", "verde"]
    for k, w in enumerate(src):
        tok(ax, k * 1.25, 0, w, *X["grey"])
    for k, w in enumerate(tgt):
        tok(ax, 0.6 + k * 1.25, 1.4, w, *X["out"])
    ax.annotate("", xy=(1.9, 1.1), xytext=(1.9, 0.3), arrowprops=dict(arrowstyle="-|>", color=D.DEC, lw=1.4))
    ax.text(2.6, 0.7, "a new sequence, of its own length and order", fontsize=10.5, color=S.GREY)
    ax.set_title("(d) a sequence to a sequence: translation", loc="left", fontsize=12)
    for ax in axs.ravel():
        ax.set_xlim(-0.8, 6.3); ax.set_ylim(-0.5, 2.0); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
