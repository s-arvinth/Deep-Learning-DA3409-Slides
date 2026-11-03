"""fits — A shallow ReLU network fitted to noisy samples of two functions

Used in: Class 5 (B), slide(s): "It works: shallow networks fit real data"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python fits.py            -> output/fits.pdf
Ported from: Class05_Shallow_NN_Regression/B_Recreated_Figures/src/figures/fig16_fits.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import L1, OUTC, GREY, relu_design

NAME = "fits"


def build():
    common.use()
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(-1, 1, 60))
    xs = np.linspace(-1, 1, 800)
    cases = [(lambda z: z ** 2,                  r"$f(x)=x^2$"),
             (lambda z: np.sin(2 * np.pi * z),   r"$f(x)=\sin(2\pi x)$")]

    fig, ax = plt.subplots(1, 2, figsize=(8.8, 4.4))
    for axis, (fn, nm), tag in zip(ax, cases, "ab"):
        y = fn(x) + 0.10 * rng.normal(size=len(x))
        cuts = np.linspace(-1, 1, 10)[1:-1]
        A = relu_design(x, cuts)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        axis.scatter(x, y, s=34, facecolor="white", edgecolor=L1, lw=1.4,
                     zorder=3, label=r"data $(x_n,y_n)$")
        axis.plot(xs, fn(xs), color=GREY, lw=1.8, ls="--", label=r"truth")
        axis.plot(xs, relu_design(xs, cuts) @ w, color=OUTC, lw=2.4,
                  label=r"network $\hat y$")
        axis.set(xlabel=r"$x$")
        axis.set_title(rf"({tag})  {nm},  $M=8$ units", fontsize=15.6)
        common.square(axis)
    ax[0].set_ylabel(r"$y$")
    h, lb = ax[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=15.0,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=2.4)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
