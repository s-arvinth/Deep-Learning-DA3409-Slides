"""What the bias correction in Adam actually corrects.

Both running averages start at zero, so for the first few dozen steps
they are far too small -- by exactly the factor 1 - beta^t. Dividing by
that factor removes the bias, and since beta < 1 the correction fades
away on its own.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S

NAME = "bias_correction"

B1, B2 = 0.9, 0.99


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    # ---- (a) the estimate of a constant gradient ---------------------
    ax = axes[0]
    T = np.arange(1, 81)
    g = 1.0
    s = np.zeros(len(T) + 1)
    for t in T:
        s[t] = B1 * s[t - 1] + (1 - B1) * g
    raw = s[1:]
    corrected = raw / (1 - B1 ** T)
    ax.axhline(g, color=S.GREY, lw=1.2, ls=(0, (3, 3)))
    ax.fill_between(T, raw, g, color=S.OUTC, alpha=0.14, zorder=1)
    ax.plot(T, raw, color=S.OUTC, lw=2.6, zorder=3,
            label=r"raw average $s^{(\tau)}$")
    ax.plot(T, corrected, color=S.L1, lw=2.6, ls=(0, (6, 2.4)), zorder=4,
            label=r"corrected $s^{(\tau)}/(1-\beta_1^{\tau})$")
    ax.annotate("the bias", xy=(11, 0.55), fontsize=14, color=S.OUTC)
    ax.annotate("the true gradient", xy=(46, 1.045), fontsize=13.5,
                color=S.GREY)
    ax.set_xlabel(r"iteration $\tau$")
    ax.set_ylabel("estimate of the gradient")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="lower right", fontsize=12.5)
    ax.set_title(r"(a) the first steps are far too small")
    S.square(ax)

    # ---- (b) the size of the correction ------------------------------
    ax = axes[1]
    T = np.arange(1, 401)
    ax.semilogx(T, 1.0 / (1 - B1 ** T), color=S.L1, lw=2.5,
                label=r"$1/(1-\beta_1^{\tau})$,  $\beta_1 = 0.9$")
    ax.semilogx(T, 1.0 / (1 - B2 ** T), color=S.OUTC, lw=2.5,
                label=r"$1/(1-\beta_2^{\tau})$,  $\beta_2 = 0.99$")
    ax.axhline(1.0, color=S.GREY, lw=1.4, ls=(0, (5, 3)))
    ax.set_xlabel(r"iteration $\tau$")
    ax.set_ylabel("correction factor")
    ax.set_ylim(0, 12)
    ax.legend(loc="upper right", fontsize=12.5)
    ax.set_title("(b) it fades away on its own")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
