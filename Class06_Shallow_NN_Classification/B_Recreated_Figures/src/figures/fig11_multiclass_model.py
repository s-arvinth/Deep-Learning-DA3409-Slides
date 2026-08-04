#!/usr/bin/env python3
"""K = 3 outputs on a scalar input: pre-activations -> softmax -> regions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import softmax, relu

NAME = "multiclass_model"

#  one shared hidden layer: the input plus three ReLU units with joints
#  at 0.25, 0.5, 0.75.  Every output reads the SAME units, so all three
#  bend at the same places.
JOINTS = np.array([0.25, 0.50, 0.75])
W2 = np.array([[ 3.0, -11.0,   9.0,   0.0, -4.0],
               [-1.0,  10.0, -13.0,  -2.0,  6.0],
               [-4.0,   0.0,   3.0,  10.0,  2.0]])


def net(x):
    z = np.vstack([x] + [relu(x - c) for c in JOINTS])
    return W2[:, 0][:, None] + W2[:, 1:] @ z


def build():
    style.use()
    x = np.linspace(0, 1, 1200)
    A = net(x)
    P = softmax(A, axis=0)
    lead = P.argmax(0)
    cols = [L1, L2, OUTC]

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))
    for k in range(3):
        ax[0].plot(x, A[k], color=cols[k], lw=2.4,
                   label=rf"$a_{k+1}(x)$")
        ax[0].axvline(JOINTS[min(k, 2)], color=GREY, lw=0.8, ls=":")
        ax[1].plot(x, P[k], color=cols[k], lw=2.4,
                   label=rf"$\hat y_{k+1}(x)$")
    ax[0].legend(loc="upper left", fontsize=13.0)
    ax[0].set(xlabel=r"$x$", ylabel=r"$a_k$")
    ax[0].set_title(r"(a)  $K=3$ piecewise-linear outputs", fontsize=15.0)

    ax[1].legend(loc="upper left", fontsize=13.0)
    ax[1].set(xlabel=r"$x$", ylabel=r"$\hat y_k$", ylim=(-0.04, 1.16))
    ax[1].set_title(r"(b)  after softmax: $\sum_k\hat y_k=1$",
                    fontsize=15.0)

    for k in range(3):
        ax[2].fill_between(x, 0, 1, where=lead == k, color=cols[k],
                           alpha=.22, lw=0)
        ax[2].plot(x, P[k], color=cols[k], lw=2.0)
    for c in JOINTS:
        ax[2].axvline(c, color=GREY, lw=0.9, ls=":")
    ax[2].plot(x, P.max(0), color="black", lw=1.4, ls="--",
               label=r"$\max_k \hat y_k$")
    ax[2].legend(loc="lower right", fontsize=13.0, framealpha=.92,
                 facecolor="white", edgecolor="none")
    ax[2].set(xlabel=r"$x$", ylabel=r"$\hat y_k$", ylim=(0, 1.0))
    ax[2].set_title(r"(c)  decision regions: $\arg\max_k \hat y_k$",
                    fontsize=15.0)
    for a_ in ax:
        style.square(a_)
    fig.suptitle(r"shared hidden units $\Rightarrow$ all three outputs "
                 r"bend at the same $x$", fontsize=16.2, y=1.03)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
