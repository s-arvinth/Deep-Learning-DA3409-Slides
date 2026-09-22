"""Training an RNN language model, in the layout of Jurafsky & Martin
(2026), Fig. 14.6: at every position the model reads the true word,
predicts a distribution over the vocabulary, and pays the cross-entropy
of the true NEXT word (Eq. 14.11); the loss is the average over the
sequence.  Teacher forcing: the input at t+1 is the true word, not the
prediction."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from common import style as S
from common import draw as D

NAME = "rnn_lm_training"
WORDS = ["So", "long", "and", "thanks", "for"]
NEXT = ["long", "and", "thanks", "for", "all"]


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    xs = [0, 1.25, 2.5, 3.75, 5.0]
    D.band(ax, -0.55, 5.55, 0.45, 1.35, D.ENC)
    prev = None
    for x, w, nx in zip(xs, WORDS, NEXT):
        c = D.column(ax, x, 0, D.ENC, D.ENC_F, rows=("emb", "hid", "soft"))
        ax.text(x, -0.85, w, ha="center", fontsize=13)
        D.arrow(ax, (x, -0.6), c["emb"]); D.arrow(ax, c["emb"], c["hid"]); D.arrow(ax, c["hid"], c["soft"])
        if prev is not None:
            D.arrow(ax, prev, c["hid"], color=D.ENC)
        prev = c["hid"]
        # the loss box and the next word
        ax.add_patch(plt.Rectangle((x - 0.42, 2.5), 0.84, 0.42, fc="#F1D6D6", ec=D.DEC, lw=1.1, zorder=3))
        ax.text(x, 2.71, "$-\\log \\hat{y}_t[w_{t+1}]$", ha="center", va="center", fontsize=9.5, color=D.DEC)
        D.arrow(ax, c["soft"], (x, 2.48), color=D.DEC)
        ax.text(x, 3.25, nx, ha="center", fontsize=13, color=D.DEC)
        D.arrow(ax, (x, 3.12), (x, 2.94), color=D.DEC)
    ax.text(5.75, 0.9, "$\\cdots$", ha="center", va="center", fontsize=15)
    for k, lab in enumerate(("input embeddings $\\mathbf{e}_t$", "RNN $\\mathbf{h}_t$",
                              "softmax over $V$: $\\hat{\\mathbf{y}}_t$")):
        ax.text(-0.75, k * 0.9, lab, ha="right", va="center", fontsize=11.5, color=S.GREY)
    ax.text(-0.75, 2.71, "loss at $t$", ha="right", va="center", fontsize=11.5, color=D.DEC)
    ax.text(-0.75, 3.25, "next word $w_{t+1}$", ha="right", va="center", fontsize=11.5, color=D.DEC)
    ax.text(6.9, 2.71, "$L = \\dfrac{1}{T}\\sum_t -\\log \\hat{y}_t[w_{t+1}]$", ha="center", va="center",
            fontsize=13, color=D.DEC)
    ax.text(6.9, 1.9, "average over the\nsequence, Eq. 14.11", ha="center", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-3.3, 8.2); ax.set_ylim(-1.2, 3.6); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
