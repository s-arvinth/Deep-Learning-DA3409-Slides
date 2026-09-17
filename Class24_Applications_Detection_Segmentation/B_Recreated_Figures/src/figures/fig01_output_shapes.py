"""Three tasks, three output structures.

The three applications of this class differ in what the network must
emit for one 224 x 224 image:

    classification   one distribution over C classes           C
    detection        a grid of S x S cells, each with B boxes    S^2 (5B + C)
                     of five numbers and one class distribution
    segmentation     one distribution per pixel                 H W C

(a) The number of output values, with the figures the books quote:
    C = 1000 for ImageNet; S = 7, B = 2, C = 20 for YOLO; 224 x 224
    pixels and C = 21 classes for the segmentation network.
(b) The same three drawn: what the output looks like.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import style as S

NAME = "output_shapes"

C_CLS, S_GRID, B_BOX, C_DET, H, C_SEG = 1000, 7, 2, 20, 224, 21


def build():
    S.use()
    sizes = [C_CLS, S_GRID ** 2 * (5 * B_BOX + C_DET), H * H * C_SEG]
    names = ["classification\n$C$", "detection\n$S^2(5B + C)$",
             "segmentation\n$H\\,W\\,C$"]
    print("output values:", dict(zip(("cls", "det", "seg"), sizes)))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.9]})
    ax = axes[0]
    cols = [S.L1, S.OUTC, S.L2]
    ax.bar(range(3), sizes, color=cols, width=0.62)
    for i, v in enumerate(sizes):
        ax.text(i, v * 1.25, "{:,}".format(v), ha="center", fontsize=13)
    ax.set_yscale("log"); ax.set_ylim(3e2, 6e6)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=12.5)
    ax.set_ylabel("output values per image")
    ax.set_title("(a) how much the network must emit")
    S.square(ax)

    ax = axes[1]
    ax.set_xlim(0, 11.8); ax.set_ylim(0.3, 4.6); ax.set_aspect("equal"); ax.axis("off")
    # classification: one bar chart
    ax.add_patch(Rectangle((0.2, 0.6), 3.2, 3.2, fc="white", ec="black", lw=1))
    h = np.array([0.15, 0.62, 0.08, 0.32, 0.12, 0.2, 0.06, 0.45])
    for i, hv in enumerate(h):
        ax.add_patch(Rectangle((0.45 + 0.35 * i, 0.75), 0.25, 2.8 * hv,
                               fc=S.L1 if i != 1 else S.OUTC, ec="none"))
    ax.text(1.8, 4.1, "one distribution", ha="center", fontsize=12.5, color=S.L1)
    # detection: grid with boxes
    ax.add_patch(Rectangle((4.2, 0.6), 3.2, 3.2, fc="white", ec="black", lw=1))
    for t in np.linspace(0, 3.2, S_GRID + 1)[1:-1]:
        ax.plot([4.2 + t, 4.2 + t], [0.6, 3.8], color=S.GREY, lw=0.5, alpha=0.6)
        ax.plot([4.2, 7.4], [0.6 + t, 0.6 + t], color=S.GREY, lw=0.5, alpha=0.6)
    for (x, y, w, hh) in ((4.7, 1.2, 1.4, 1.6), (6.0, 2.2, 1.1, 1.2)):
        ax.add_patch(Rectangle((x, y), w, hh, fc="none", ec=S.OUTC, lw=2))
    ax.text(5.8, 4.1, "boxes on a grid", ha="center", fontsize=12.5, color=S.OUTC)
    # segmentation: pixel map
    ax.add_patch(Rectangle((8.2, 0.6), 3.2, 3.2, fc="white", ec="black", lw=1))
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:16, 0:16]
    lab = ((xx - 5) ** 2 + (yy - 9) ** 2 < 14).astype(int) + 2 * ((xx > 10) & (yy < 7))
    tint = {0: "#EFEFEF", 1: S.L2, 2: S.ACC, 3: S.ACC}
    for i in range(16):
        for j in range(16):
            ax.add_patch(Rectangle((8.2 + 0.2 * j, 0.6 + 0.2 * (15 - i)), 0.2, 0.2,
                                   fc=tint[int(lab[i, j])], ec="none"))
    ax.text(9.8, 4.1, "a label per pixel", ha="center", fontsize=12.5, color=S.L2)
    ax.set_title("(b) what the output looks like")
    fig.tight_layout(w_pad=1.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
