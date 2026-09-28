"""Carrying one fact across a sequence: RNN, LSTM, GRU.

The shared experiment: three cells of the same width, trained the same
way to read a sequence of symbols and emit the first one at the end.
(a) Training loss.  (b) Accuracy against sequence length, tested to
length 60 after training on 4-24.  The gated cells are perfect at every
length tried; the simple RNN never gets past three in four even inside
its training range.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.experiment import run, LENS_TEST

NAME = "long_dependency"


def build():
    S.use()
    R = run()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    for kind, col in (("RNN", S.OUTC), ("LSTM", S.L1), ("GRU", S.L2)):
        h = np.array(R[kind]["hist"]); k = 50
        ax.semilogy(np.arange(len(h) - k + 1) + k, np.convolve(h, np.ones(k) / k, "valid"), color=col, lw=2.2, label=kind)
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy"); ax.legend(fontsize=11)
    ax.set_title("(a) training loss"); S.square(ax)
    ax = axes[1]
    for kind, col in (("RNN", S.OUTC), ("LSTM", S.L1), ("GRU", S.L2)):
        acc = R[kind]["acc"]
        ax.plot(LENS_TEST, [acc[n] for n in LENS_TEST], "o-", color=col, lw=2.2, ms=5, label=kind)
        print("%s: accuracy at 8, 24, 60: %.2f %.2f %.2f" % (kind, acc[8], acc[24], acc[60]))
    ax.axvspan(4, 24, color=S.GREY, alpha=0.10, lw=0)
    ax.text(14, 0.08, "training lengths", ha="center", fontsize=11, color=S.GREY)
    ax.axhline(1 / 8, color=S.GREY, lw=0.9, ls=(0, (3, 3)))
    ax.set_xlabel("sequence length $T$"); ax.set_ylabel("accuracy on the first symbol"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="center right"); ax.set_title("(b) recall by length")
    S.square(ax)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
