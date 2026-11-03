"""step_activation — the step activation, and sigmoids sharpening towards it

Used in: Class 2 (B), slide(s): "The activation is a step function"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python step_activation.py            -> output/step_activation.pdf
"""
import numpy as np
import matplotlib.pyplot as plt

from common import L1, L2, OUTC, GREY, save

NAME = "step_activation"


def build():
    z = np.linspace(-3, 3, 2000)
    theta = 0.8
    step = (z >= theta).astype(float)

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 2.9))

    ax[0].plot(z, step, color=OUTC, lw=1.8)
    ax[0].axvline(theta, color=GREY, ls=":", lw=0.8)
    ax[0].text(theta + 0.08, 0.45, r"$\theta$", color=GREY, fontsize=11)
    ax[0].set(xlabel=r"total excitation $z$", ylabel="output",
              ylim=(-0.15, 1.2),
              title=r"(a)  step: fires iff $z \geq \theta$")

    for k, c in zip([1, 4, 16], [L1, L2, OUTC]):
        ax[1].plot(z, 1 / (1 + np.exp(-k * (z - theta))), color=c,
                   lw=1.5, label=rf"$k={k}$")
    ax[1].plot(z, step, color=GREY, lw=1.0, ls="--", label="step")
    ax[1].axvline(theta, color=GREY, ls=":", lw=0.8)
    ax[1].legend(loc="upper left")
    ax[1].set(xlabel=r"$z$", ylabel="output", ylim=(-0.15, 1.2),
              title=r"(b)  $\sigma(k(z-\theta))$ sharpens to the step")
    fig.tight_layout(w_pad=2.0)
    return save(fig, NAME)


if __name__ == "__main__":
    build()
