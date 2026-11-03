"""action_potential — membrane potential trace: sub-threshold bumps, one spike, refractory dip

Used in: Class 2 (B), slide(s): "Firing: an all-or-nothing threshold"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python action_potential.py            -> output/action_potential.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, L2, OUTC, GREY, save

NAME = "action_potential"


def build():
    t = np.linspace(0, 10, 2000)
    v = np.full_like(t, -70.0)                       # resting, mV
    # sub-threshold bumps that fail to fire
    for c in (1.2, 2.4):
        v += 6 * np.exp(-((t - c) ** 2) / 0.05)
    # the spike itself
    spike = 100 * np.exp(-((t - 5.0) ** 2) / 0.010)
    after = -18 * np.exp(-((t - 5.35) ** 2) / 0.050)  # refractory dip
    v = v + spike + after

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axhline(-55, color=GREY, ls="--", lw=1.0)
    ax.text(9.6, -53, "threshold", color=GREY, ha="right", fontsize=9)
    ax.axhline(-70, color=GREY, ls=":", lw=0.8)
    ax.text(0.1, -68.5, "resting", color=GREY, fontsize=9)
    ax.plot(t, v, color=OUTC, lw=1.6)
    ax.annotate("sub-threshold:\nno spike", xy=(2.4, -62), xytext=(2.9, -30),
                fontsize=9, color=L1,
                arrowprops=dict(arrowstyle="->", color=L1, lw=0.8))
    ax.annotate("action potential", xy=(5.0, 22), xytext=(6.1, 8),
                fontsize=9, color=OUTC,
                arrowprops=dict(arrowstyle="->", color=OUTC, lw=0.8))
    ax.annotate("refractory dip", xy=(5.45, -84), xytext=(6.6, -95),
                fontsize=9, color=L2,
                arrowprops=dict(arrowstyle="->", color=L2, lw=0.8))
    ax.set(xlabel="time (ms)", ylabel="membrane potential (mV)",
           ylim=(-105, 45), xlim=(0, 10))
    return save(fig, NAME)


if __name__ == "__main__":
    build()
