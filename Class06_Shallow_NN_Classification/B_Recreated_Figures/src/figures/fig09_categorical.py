#!/usr/bin/env python3
"""The categorical distribution and the softmax that produces it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import softmax

NAME = "categorical"


def build():
    style.use()
    A = np.array([[2.1, 0.4, -0.9, 0.2, -1.6],
                  [0.3, 0.5, 0.2, 0.4, 0.1],
                  [-1.0, -0.4, 3.0, 0.1, -0.7]])
    K = A.shape[1]
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 4.0))
    for a_, av, c in zip(ax, A, [L1, L2, OUTC]):
        p = softmax(av)
        a_.bar(np.arange(1, K + 1), p, width=0.55, color=c, alpha=.85,
               edgecolor="white", lw=1.2)
        for k, h in enumerate(p, start=1):
            a_.text(k, h + 0.02, f"{h:.2f}", ha="center", fontsize=13.0,
                    color=c)
        a_.set(xticks=np.arange(1, K + 1), ylim=(0, 1.0), xlabel=r"class $k$")
        a_.set_title(r"$\mathbf{a}=[$" +
                     ", ".join(f"{v:.1f}" for v in av) + r"$]$",
                     fontsize=13.7)
        style.square(a_)
    ax[0].set_ylabel(r"$\hat y_k$")
    fig.suptitle(r"$\hat y_k=\dfrac{\exp(a_k)}{\sum_j \exp(a_j)}$:  "
                 r"non-negative, sums to one", fontsize=16.9, y=1.06)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
