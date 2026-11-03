"""Gabor filters, the model of a simple cell.

Bishop & Bishop (2024), Eqs. 10.6-10.8, drawn in the layout of Fig.
10.11: the orientation theta runs from 0 in the top row to pi/2 in the
bottom row, the frequency omega from 1 in the left column to 10 in the
right.  Every panel is the formula evaluated on a grid; nothing else.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import gabor

NAME = "gabor"
THETAS = np.linspace(0, np.pi / 2, 4)
OMEGAS = np.array([1.0, 4.0, 7.0, 10.0])


def build():
    S.use()
    fig, axes = plt.subplots(4, 4, figsize=(5.2, 5.2))
    for i, th in enumerate(THETAS):
        for j, om in enumerate(OMEGAS):
            G = gabor(48, th, om, alpha=1.2, beta=1.2)
            axes[i, j].imshow(G, cmap="gray", vmin=-1, vmax=1)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    for j, om in enumerate(OMEGAS):
        axes[0, j].set_title("$\\omega = %g$" % om, fontsize=11)
    for i, th in enumerate(THETAS):
        axes[i, 0].set_ylabel(["$\\theta = 0$", "$\\theta = \\pi/6$",
                               "$\\theta = \\pi/3$", "$\\theta = \\pi/2$"][i], fontsize=11)
    fig.tight_layout(w_pad=0.3, h_pad=0.3)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
