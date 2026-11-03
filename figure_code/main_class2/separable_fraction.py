"""separable_fraction — how many Boolean functions of n inputs are linearly separable

Used in: Class 2 (B), slide(s): "How rare is linear separability?"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python separable_fraction.py            -> output/separable_fraction.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, OUTC, GREY, save

NAME = "separable_fraction"


def build():
    n    = np.array([1, 2, 3, 4, 5])
    tot  = np.array([4, 16, 256, 65536, 4294967296], dtype=float)
    sep  = np.array([4, 14, 104, 1882, 94572], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.1))
    w = 0.36
    ax[0].bar(n - w/2, tot, w, color=L1, label="all Boolean functions")
    ax[0].bar(n + w/2, sep, w, color=OUTC, label="linearly separable")
    ax[0].set(yscale="log", xlabel=r"number of inputs $n$",
              ylabel="count", xticks=n,
              title="(a)  how many functions")
    ax[0].legend(loc="upper left")

    frac = 100 * sep / tot
    ax[1].plot(n, frac, color=OUTC, marker="o", ms=6, mew=0.8, mec="white")
    for xi, yi in zip(n, frac):
        dx, ha = (-10, "right") if xi == n[-1] else (0, "center")
        ax[1].annotate(f"{yi:.3g}%", (xi, yi), textcoords="offset points",
                       xytext=(dx, 9), ha=ha, fontsize=8.5, color=GREY)
    ax[1].set_xlim(0.6, 5.4)
    ax[1].set(yscale="log", xlabel=r"number of inputs $n$",
              ylabel="separable (\\% of all)", xticks=n,
              title="(b)  the fraction collapses")
    fig.tight_layout(w_pad=2.0)
    return save(fig, NAME)


if __name__ == "__main__":
    build()
