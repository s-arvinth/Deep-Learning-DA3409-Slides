"""approximation — least-squares fits of 3, 10 and 40 threshold units to a target curve

Used in: Class 2 (B), slide(s): "Simple units add up to complicated functions"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python approximation.py            -> output/approximation.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, L2, OUTC, GREY, save

NAME = "approximation"


def build():
    x = np.linspace(-1, 1, 1500)
    target = np.sin(2.4 * np.pi * x) * np.exp(-x**2)

    def fit_steps(m):
        """least-squares fit of m shifted step (threshold) units"""
        cuts = np.linspace(-1, 1, m + 2)[1:-1]
        A = np.stack([(x >= c).astype(float) for c in cuts] +
                     [np.ones_like(x)], 1)
        w, *_ = np.linalg.lstsq(A, target, rcond=None)
        return A @ w

    fig, ax = plt.subplots(1, 3, figsize=(9.8, 2.8))
    for a, m, c in zip(ax, [3, 10, 40], [L1, L2, OUTC]):
        a.plot(x, target, color=GREY, lw=1.6, ls="--", label="target")
        a.plot(x, fit_steps(m), color=c, lw=1.5, label=f"{m} units")
        a.legend(loc="upper right", fontsize=8.5)
        a.set(xlabel=r"$x$", ylabel=r"$f(x)$",
              title=rf"({'abc'[[3,10,40].index(m)]})  {m} threshold units")
    fig.tight_layout(w_pad=1.7)
    return save(fig, NAME)


if __name__ == "__main__":
    build()
