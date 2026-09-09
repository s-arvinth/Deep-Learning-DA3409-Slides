"""Changing resolution, in both directions.

(a) One signal, reduced by max pooling with window two and by a strided
    convolution. Both halve the length; they differ in what they keep.

(b) The same reduced signal put back to full length three ways:
    duplication, zero-insertion, and linear interpolation. None of them
    recovers what pooling discarded, which is why an architecture that
    needs the detail keeps a copy of it instead.

(c) Why pooling buys invariance. The input is translated by 0 to 8
    positions and the representation compared with the untranslated one,
    each normalised by its own scale so the four curves are comparable.
    A shift of one changes the raw signal by 14% and an 8-wide pooled
    signal by 6%. The wider the window, the more shift it absorbs --
    and the less it can say about where anything is.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import conv1d, maxpool1d, upsample1d

NAME = "resolution"

N = 64
K = np.array([0.25, 0.5, 0.25])


def _sig(n, rng):
    t = np.linspace(0, 1, n)
    return (np.exp(-0.5 * ((t - 0.30) / 0.05) ** 2)
            + 0.75 * np.exp(-0.5 * ((t - 0.62) / 0.03) ** 2)
            + 0.35 * (t > 0.80) + 0.03 * rng.normal(size=n))


def build():
    S.use()
    rng = np.random.default_rng(2)
    x = _sig(N + 16, rng)[8:8 + N]

    pooled = maxpool1d(x, 2)
    strided = conv1d(x, K, s=2, p=1)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))

    ax = axes[0]
    ax.plot(np.arange(N), x, color=S.GREY, lw=1.8, label="input, $n = %d$" % N)
    ax.plot(np.arange(len(pooled)) * 2, pooled, "o-", color=S.OUTC, lw=2.2,
            ms=4.2, label="max pool, $k=2$")
    ax.plot(np.arange(len(strided)) * 2, strided, "o-", color=S.L2, lw=2.2,
            ms=4.2, label="strided conv, $s=2$")
    ax.legend(loc="upper left", fontsize=11.5)
    ax.set_title("(a) two ways down")

    ax = axes[1]
    ax.plot(np.arange(N), x, color=S.GREY, lw=1.8, label="original")
    for mode, col, lab in (("duplicate", S.OUTC, "duplicate"),
                           ("zeros", S.ACC, "insert zeros"),
                           ("linear", S.L2, "linear")):
        u = upsample1d(pooled, 2, mode)
        ax.plot(np.arange(len(u)), u, color=col, lw=1.9, alpha=0.95,
                label=lab)
    ax.legend(loc="upper left", fontsize=11.5)
    ax.set_title("(b) and three ways back up")

    ax = axes[2]
    shifts = np.arange(0, 9)
    long = _sig(N + 64, rng)
    x0 = long[32:32 + N]
    ref = np.mean(np.abs(x0))
    raw = [np.mean(np.abs(long[32 - t:32 - t + N] - x0)) / ref
           for t in shifts]
    ax.plot(shifts, raw, "o-", color=S.GREY, lw=2.3, ms=6,
            label="no pooling")
    cols = plt.get_cmap(S.CAT)(np.linspace(0.10, 0.74, 3))
    for K_, col in zip((2, 4, 8), cols):
        p0 = maxpool1d(x0, K_)
        r0 = np.mean(np.abs(p0))
        d = [np.mean(np.abs(maxpool1d(long[32 - t:32 - t + N], K_) - p0)) / r0
             for t in shifts]
        ax.plot(shifts, d, "o-", color=col, lw=2.3, ms=5.5,
                label=r"max pool, $k = %d$" % K_)
    ax.set_xlabel("shift of the input (positions)")
    ax.set_ylabel("relative change in the representation")
    ax.legend(loc="upper left", fontsize=11.5)
    ax.set_title("(c) a wider window absorbs a larger shift")

    for ax in axes[:2]:
        ax.set_xlabel("position $i$")
        ax.set_ylabel("value")
    for ax in axes:
        ax.set_box_aspect(0.72)

    fig.tight_layout(w_pad=1.4)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
