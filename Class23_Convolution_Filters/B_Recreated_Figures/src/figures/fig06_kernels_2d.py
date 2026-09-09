"""What a 3x3 kernel does to a picture.

The scene is constructed, not photographed: a disc, a bar, a bright
strip and a gradient, so every pixel in the deck is reproducible from
`common/models.py`.

Each panel applies one 3x3 kernel with zero padding. The same three
numbers per row are reused at every position, which is the whole content
of weight sharing, and the outputs show what "feature detector" means:
the edge kernels respond only where the intensity changes, and in the
direction they were built for.

The number under each panel is the kernel's response to the patch that
maximises it, normalised -- by Cauchy-Schwarz that patch is a multiple
of the kernel itself, which is Bishop & Bishop (2024), Exercise 10.1.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import synthetic_image, conv2d, KERNELS

NAME = "kernels_2d"

SHOW = ("blur", "vertical edge", "horizontal edge", "sharpen")


def build():
    S.use()
    img = synthetic_image(72)
    fig, axes = plt.subplots(1, 5, figsize=(15.4, 3.9))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("the image")
    axes[0].annotate("constructed, not photographed",
                     xy=(0.5, -0.09), xycoords="axes fraction", ha="center",
                     va="top", fontsize=11.5, color=S.GREY)

    for ax, name in zip(axes[1:], SHOW):
        W = KERNELS[name]
        out = conv2d(img, W)
        v = np.max(np.abs(out))
        ax.imshow(out, cmap="RdBu_r", vmin=-v, vmax=v,
                  interpolation="nearest")
        ax.set_title(name)
        ax.annotate(r"$\|\mathbf{w}\| = %.2f$" % np.linalg.norm(W),
                    xy=(0.5, -0.09), xycoords="axes fraction", ha="center",
                    va="top", fontsize=11.5, color=S.GREY)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(w_pad=1.1)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
