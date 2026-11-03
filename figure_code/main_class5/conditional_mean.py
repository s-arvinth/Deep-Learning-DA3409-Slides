"""conditional_mean — Gaussian noise about a linear mean; the optimal prediction is E[y|x]

Used in: Class 5 (B), slide(s): "Why least squares? The probabilistic view"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python conditional_mean.py            -> output/conditional_mean.pdf
Ported from: Class05_Shallow_NN_Regression/B_Recreated_Figures/src/figures/fig04_conditional_mean.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import L1, L2, OUTC, GREY, gaussian_cloud

NAME = "conditional_mean"


def build():
    common.use()
    xd, yd, (w0, w1, sd) = gaussian_cloud()
    xs = np.linspace(0, 1, 400)
    mean = w0 + w1 * xs

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.scatter(xd, yd, s=20, color=L1, alpha=.42, lw=0, label=r"data")
    for k, al in [(1, .26), (2, .14)]:
        ax.fill_between(xs, mean - k * sd, mean + k * sd, color=OUTC,
                        alpha=al, lw=0)
    ax.plot(xs, mean, color=OUTC, lw=2.6,
            label=r"$\mathbb{E}[y\mid x]$")

    x0 = 0.58
    yy = np.linspace(mean[0] - 0.35, mean[-1] + 0.55, 400)
    pdf = np.exp(-((yy - (w0 + w1 * x0)) ** 2) / (2 * sd ** 2))
    ax.fill_betweenx(yy, x0, x0 + 0.26 * pdf, color=L2, alpha=.20, lw=0)
    ax.plot(x0 + 0.26 * pdf, yy, color=L2, lw=2.2)
    ax.axvline(x0, color=GREY, lw=0.9, ls=":")
    ax.annotate(r"$p(y\mid x_0)$", xy=(x0 + 0.29, w0 + w1 * x0 + 0.02),
                color=L2, fontsize=19.5, ha="left", va="center")
    ax.annotate(r"$=\mathcal{N}(y\mid\hat y(x_0),\,\sigma^2)$",
                xy=(x0 + 0.29, w0 + w1 * x0 - 0.15), color=L2, fontsize=15.0,
                ha="left", va="center")
    ax.annotate(r"$x_0$", xy=(x0, ax.get_ylim()[0]), xytext=(x0 + 0.012, 0.02),
                color=GREY, fontsize=15.6)
    ax.legend(loc="upper left", fontsize=15.0)
    ax.set(xlabel=r"input  $x$", ylabel=r"target  $y$", xlim=(0, 1.42),
           title=r"the best prediction is the conditional mean")
    ax.title.set_fontsize(12.5)
    common.square(ax)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
