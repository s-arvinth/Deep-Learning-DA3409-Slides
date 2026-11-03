"""weights_bias — weights tilt the boundary; the bias shifts it

Used in: Class 3 (B), slide(s): "What the weights and the bias each control"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python weights_bias.py            -> output/weights_bias.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, L2, OUTC, GREY, boundary, save

NAME = "weights_bias"


def build():
    xs = np.linspace(-3, 3, 20)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))

    # (a) same threshold, different weight vectors -> different tilt
    for w, c, lab in [((0, 1, 1), L1, r"$w=(1,1)$"),
                      ((0, 2, 0.6), L2, r"$w=(2,0.6)$"),
                      ((0, 0.5, 2), OUTC, r"$w=(0.5,2)$")]:
        boundary(ax[0], w, xs, color=c, lw=1.7, label=lab)
        ax[0].arrow(0, 0, w[1] * .38, w[2] * .38, color=c, lw=1.2,
                    head_width=0.12, length_includes_head=True, alpha=.85)
    ax[0].set(xlim=(-3, 3), ylim=(-3, 3), xlabel=r"$x_1$", ylabel=r"$x_2$",
              title="(a)  weights set the orientation")
    ax[0].legend(loc="lower right", fontsize=8.5)

    # (b) same weights, different bias -> parallel shift
    for b, c in zip([-2.0, 0.0, 2.0], [L1, L2, OUTC]):
        boundary(ax[1], (b, 1, 1), xs, color=c, lw=1.7,
                 label=rf"$w_0={b:+.0f}$")
    ax[1].set(xlim=(-3, 3), ylim=(-3, 3), xlabel=r"$x_1$", ylabel=r"$x_2$",
              title="(b)  the bias sets the offset")
    ax[1].legend(loc="lower right", fontsize=8.5)
    for a in ax:
        a.set_aspect("equal", "box")
        a.axhline(0, color=GREY, lw=0.4, ls=":")
        a.axvline(0, color=GREY, lw=0.4, ls=":")
    fig.tight_layout(w_pad=2.0)
    return save(fig, NAME)


if __name__ == "__main__":
    build()
