"""What pooling discards, and what the decoder cannot recover alone.

The scene of Class 23 is average-pooled L times and brought back with
bilinear up-sampling L times, for L = 1, 2, 3, 4.  The root-mean-square
error against the original is printed on each panel.  Edges blur and
the thin line vanishes: whatever the bottleneck keeps, it does not
keep WHERE things were to the pixel.  That is the information a U-net
skip path hands the decoder directly.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import synthetic_image, avgpool, bilinear_up

NAME = "downup"


def build():
    S.use()
    img = synthetic_image(64)
    fig, axes = plt.subplots(1, 5, figsize=(11.0, 2.9))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original, $64^2$", fontsize=12, pad=6)
    errs = []
    for L, ax in zip((1, 2, 3, 4), axes[1:]):
        y = img.copy()
        for _ in range(L):
            y = avgpool(y)
        side = y.shape[0]
        for _ in range(L):
            y = bilinear_up(y)
        e = float(np.sqrt(np.mean((y - img) ** 2)))
        errs.append(e)
        ax.imshow(y, cmap="gray", vmin=0, vmax=1)
        ax.set_title("down to $%d^2$ and back" % side, fontsize=12, pad=6)
        ax.set_xlabel("RMS error %.3f" % e, fontsize=12, color=S.OUTC, labelpad=6)
    print("RMS error after L = 1..4 levels:", np.round(errs, 3))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(w_pad=1.2)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
