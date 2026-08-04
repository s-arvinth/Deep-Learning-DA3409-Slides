#!/usr/bin/env python3
"""Binary classification end to end: network output -> sigmoid -> likelihood."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.models import sigmoid, relu

NAME = "binary_model"

#  three hidden units with joints at x = 0.35 and 0.72, plus the input
#  itself: a(x) is piecewise linear and crosses zero twice
JOINTS = np.array([0.35, 0.72])
W2 = np.array([-3.2, 14.0, -26.0, 22.0])       # [bias, x, relu1, relu2]


def net(x):
    z = np.vstack([x] + [relu(x - c) for c in JOINTS])
    return W2[0] + (W2[1:][:, None] * z).sum(0)


def build():
    style.use()
    rng = np.random.default_rng(6)
    x = np.linspace(0, 1, 900)
    a = net(x)
    yhat = sigmoid(a)

    xn = np.sort(rng.uniform(0, 1, 22))
    yn = (rng.uniform(size=len(xn)) < sigmoid(net(xn))).astype(float)
    pn = sigmoid(net(xn))
    lik = np.where(yn == 1, pn, 1 - pn)

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))

    ax[0].plot(x, a, color=OUTC, lw=2.6)
    ax[0].axhline(0, color=GREY, lw=0.9, ls=":")
    ax[0].set(xlabel=r"$x$", ylabel=r"$a(x)$")
    ax[0].set_title(r"(a)  network output $a(x)$: piecewise linear",
                    fontsize=15.0)

    ax[1].plot(x, yhat, color=L2, lw=2.8)
    ax[1].axhline(0.5, color=GREY, lw=0.9, ls=":")
    ax[1].scatter(xn, yn, s=42, facecolor="white", edgecolor=L1, lw=1.3,
                  zorder=4, label=r"data $(x_n,y_n)$")
    ax[1].legend(loc="center left", fontsize=13.0, framealpha=.92,
                 facecolor="white", edgecolor="none")
    ax[1].set(xlabel=r"$x$", ylabel=r"$\hat y(x)$", ylim=(-0.06, 1.08))
    ax[1].set_title(r"(b)  $\hat y=\sigma(a)$: a probability",
                    fontsize=15.0)

    ax[2].vlines(xn, 0, lik, color=GREY, lw=1.0, ls=":")
    ax[2].scatter(xn, lik, s=46, c=[OUTC if v else L1 for v in yn],
                  edgecolor="white", lw=0.9, zorder=4)
    ax[2].set(xlabel=r"$x$", ylabel="likelihood of the observed label",
              ylim=(0, 1.08))
    ax[2].set_title(r"(c)  $\hat y_n^{\,y_n}(1-\hat y_n)^{1-y_n}$",
                    fontsize=15.0)
    for a_ in ax:
        style.square(a_)
    fig.tight_layout(w_pad=2.2)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
