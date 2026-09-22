"""The bottleneck, measured.

Two encoder-decoders of the same size (d = 48), trained the same way
on the reversal task for sources of length 2-10: one with the fixed
context c = h^e_n of Jurafsky Eq. 14.33, one with the dot-product
attention of Eqs. 14.35-14.37.

(a) Training loss.  (b) Exact-match accuracy of the greedy decoding by
    source length, tested to length 16.  The fixed context degrades
    with length even inside the training range, because everything
    about the source must pass through one vector; attention holds and
    degrades more slowly beyond the range it was trained on.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.experiment import run, LENS_TEST

NAME = "bottleneck"


def build():
    S.use()
    R = run()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    for name, col, lab in (("fixed", S.OUTC, "fixed context $\\mathbf{c} = \\mathbf{h}^e_n$"),
                           ("attention", S.L2, "attention")):
        h = np.array(R[name]["hist"]); k = 50
        sm = np.convolve(h, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + k, sm, color=col, lw=2.2, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy per token")
    ax.legend(fontsize=11); ax.set_title("(a) training loss, same model size")
    S.square(ax)
    ax = axes[1]
    for name, col, lab in (("fixed", S.OUTC, "fixed context"), ("attention", S.L2, "attention")):
        acc = R[name]["acc"]
        ax.plot(LENS_TEST, [acc[n] for n in LENS_TEST], "o-", color=col, lw=2.2, ms=5, label=lab)
        print("%s: exact match at n = 8, 10, 12: %.2f, %.2f, %.2f" % (name, acc[8], acc[10], acc[12]))
    ax.axvspan(2, 10, color=S.GREY, alpha=0.10, lw=0)
    ax.text(6, 0.06, "training lengths", ha="center", fontsize=11, color=S.GREY)
    ax.set_xlabel("source length $n$"); ax.set_ylabel("exact-match accuracy"); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11, loc="upper right"); ax.set_title("(b) the whole output right, by length")
    S.square(ax)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
