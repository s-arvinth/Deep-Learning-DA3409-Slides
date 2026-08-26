"""The three initialization rules, and what they do at depth fifty.

(a) the prescribed variance against the fan-in of the layer. The factor
of two in He initialization is exactly the half of the distribution
that ReLU discards.

(b) the distribution of pre-activations at layer 50 under each rule,
for a layer of 100 units. Only the rule matched to the activation
function keeps the spread where it started.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import (he_variance, glorot_variance, lecun_variance,
                           relu)

NAME = "init_rules"

D, K = 100, 50


def _final_layer(sigma2, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, D))
    for _ in range(K):
        W = rng.normal(0.0, np.sqrt(sigma2), size=(D, D))
        a = relu(a) @ W.T
    return a.ravel()


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    fan = np.geomspace(4, 4096, 200)
    ax.loglog(fan, he_variance(fan), color=S.OUTC, lw=2.6,
              label=r"He:  $2/D$")
    ax.loglog(fan, glorot_variance(fan, fan), color=S.L2, lw=2.6,
              ls=(0, (6, 2.5)), label=r"Glorot:  $4/(D + D')$")
    ax.loglog(fan, lecun_variance(fan), color=S.L1, lw=2.6,
              label=r"LeCun:  $1/D$")
    ax.set_xlabel("fan-in $D$")
    ax.set_ylabel(r"initialization variance $\sigma^2_w$")
    ax.legend(loc="upper right", fontsize=13)
    ax.set_title("(a) the three rules")
    S.square(ax)

    ax = axes[1]
    cases = [(he_variance(D), S.OUTC, r"He,  $\sigma^2_w = 2/D$"),
             (lecun_variance(D), S.L1, r"LeCun,  $1/D$"),
             (4.0 / D, S.ACC, r"too large,  $4/D$")]
    for s2, col, lab in cases:
        v = _final_layer(s2)
        sd = v.std()
        ax.hist(v / max(sd, 1e-30), bins=90, range=(-4, 4), density=True,
                histtype="step", lw=2.2, color=col,
                label=lab + r"  ($\sigma = 10^{%.0f}$)"
                % np.log10(max(sd, 1e-300)))
    ax.set_xlabel("pre-activation at layer 50, rescaled by its own s.d.")
    ax.set_ylabel("density")
    ax.set_xlim(-4, 4)
    ax.annotate("the shape is the same;\nonly the scale differs,\n"
                "and it differs by $10^{16}$",
                xy=(-3.7, 0.30), fontsize=13, color=S.GREY, va="top")
    ax.legend(loc="upper right", fontsize=11.5)
    ax.set_title("(b) all three coincide once rescaled")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
