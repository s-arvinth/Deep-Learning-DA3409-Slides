"""How far back one output unit can see.

Following the construction of Prince (2023), Fig. 10.6.

(a) The span of the input that reaches one unit, after one, two, three
    and four layers of kernel size three and stride one. Each layer adds
    k - 1 = 2 positions, so the span is 3, 5, 7, 9.

(b) The same quantity against depth, for three stride schedules. The
    lines are the recursion

        r_l = r_{l-1} + (k_l - 1) * prod_{j<l} s_j ,

    and the markers are measured by pushing a single one back through
    the stack and counting the input positions it reaches. They agree
    exactly at every depth, which is the check that the recursion is
    right rather than plausible.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import rf_size, measured_rf

NAME = "receptive_field"

L = 8
SCHEDULES = ((3, 1, "all strides $1$", None),
             (3, 2, "stride $2$ every layer", None),
             (5, 1, "kernel $5$, stride $1$", None))


def build():
    S.use()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))

    # ---- (a) the span, drawn -----------------------------------------
    ax = axes[0]
    n = 13
    ks = [3, 3, 3, 3]
    r = rf_size(ks)
    cols = plt.get_cmap(S.CAT)(np.linspace(0.08, 0.76, len(ks)))
    ax.plot(np.arange(n), np.zeros(n), "o", ms=8, mfc="white", mec=S.GREY,
            mew=1.5, zorder=3)
    c = n // 2
    for l, (rl, col) in enumerate(zip(r, cols)):
        half = (rl - 1) // 2
        y = -(l + 1) * 0.55
        ax.plot([c - half, c + half], [y, y], color=col, lw=4.2,
                solid_capstyle="round", zorder=4)
        ax.plot([c, c], [y, y + 0.55], color=col, lw=1.0,
                ls=(0, (2, 2)), zorder=2)
        ax.annotate(r"after %d layers:  $r = %d$" % (l + 1, rl),
                    xy=(n + 0.4, y), color=col, fontsize=13, va="center")
    ax.set_xlim(-0.8, n + 5.6)
    ax.set_ylim(-2.6, 0.45)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.annotate("input positions", xy=(0, 0.22), color=S.GREY, fontsize=12.5)
    ax.set_title(r"(a) kernel $3$, stride $1$")
    ax.set_box_aspect(0.70)

    # ---- (b) formula against measurement ------------------------------
    ax = axes[1]
    cols = plt.get_cmap(S.CAT)(np.linspace(0.08, 0.76, len(SCHEDULES)))
    for (k, s, lab, _), col in zip(SCHEDULES, cols):
        ks = [k] * L
        ss = [s] * L
        form = rf_size(ks, ss)
        meas = measured_rf(ks, ss)
        d = np.arange(1, L + 1)
        ax.semilogy(d, form, "-", color=col, lw=2.5, label=lab)
        ax.semilogy(d, meas, "o", ms=6.5, mfc="none", mec=col, mew=1.8)
        assert form == meas, (form, meas)
    ax.plot([], [], "o", ms=6.5, mfc="none", mec=S.GREY, mew=1.8,
            label="measured")
    ax.set_xlabel("layers")
    ax.set_ylabel("receptive field (input positions)")
    ax.legend(loc="upper left", fontsize=12)
    ax.set_title("(b) the recursion, checked")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
