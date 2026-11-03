"""two_inputs — D = 2 inputs: every hidden unit is a hinge over the input plane

Used in: Class 5 (B), slide(s): "Each unit is a hinge over the input plane"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python two_inputs.py            -> output/two_inputs.pdf
Ported from: Class05_Shallow_NN_Regression/B_Recreated_Figures/src/figures/fig12_two_inputs.py
"""
import matplotlib.pyplot as plt
import numpy as np

import common
from common import shallow2d, relu, V1

NAME = "two_inputs"


def build():
    common.use()
    g = np.linspace(-2, 2, 600)
    X1, X2 = np.meshgrid(g, g)
    a, _ = shallow2d(X1, X2)

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))
    for j in range(3):
        m = ax[j].pcolormesh(X1, X2, relu(a[j]), cmap=common.SEQ,
                             shading="auto")
        m.set_rasterized(True)
        cb = fig.colorbar(m, ax=ax[j], fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=12.3)
        cb.set_label(rf"$z_{j+1}$", fontsize=14.3)
        w1, w2, b = V1[j]
        if abs(w2) > 1e-9:
            ax[j].plot(g, -(w1 * g + b) / w2, color="white", lw=2.0)
        ax[j].set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r"$x_1$")
        ax[j].set_title(rf"$z_{j+1}=h({w1:+.1f}x_1{w2:+.1f}x_2{b:+.2f})$",
                        fontsize=15.0)
        ax[j].set_aspect("equal", "box")
    ax[0].set_ylabel(r"$x_2$")
    fig.suptitle(r"white line: $a_j=0$, where the unit switches on",
                 fontsize=16.2, y=1.02)
    fig.tight_layout(w_pad=2.0)
    return common.save(fig, NAME)


if __name__ == "__main__":
    build()
