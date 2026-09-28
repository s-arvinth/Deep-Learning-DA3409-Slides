"""The leaky unit: a running average with a time constant.

Goodfellow Section 10.9.2: mu_t = alpha mu_{t-1} + (1 - alpha) v_t.
The impulse response for three values of alpha, with the time constant
1/(1 - alpha) marked: near one the unit remembers for a long time, near
zero it forgets at once.  A gate makes alpha depend on the input.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import leaky_unit

NAME = "leaky_unit"


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for a, col in ((0.5, S.OUTC), (0.9, S.L2), (0.98, S.L1)):
        mu = leaky_unit(a)
        ax.plot(mu / mu.max(), color=col, lw=2.4, label="$\\alpha = %.2f$, time constant $%.0f$" % (a, 1 / (1 - a)))
        tau = 5 + 1 / (1 - a)
        ax.axvline(tau, color=col, lw=0.9, ls=(0, (3, 3)))
    ax.axvline(5, color=S.GREY, lw=1.0); ax.text(5.6, 0.95, "impulse at $t = 5$", fontsize=11, color=S.GREY)
    ax.set_xlabel("$t$"); ax.set_ylabel("$\\mu_t$, normalised"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10.5)
    ax.set_title("a linear self-connection of weight $\\alpha$")
    fig.tight_layout()
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
