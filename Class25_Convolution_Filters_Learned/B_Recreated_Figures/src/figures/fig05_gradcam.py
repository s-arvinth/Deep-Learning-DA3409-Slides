"""Grad-CAM on a network small enough to differentiate by hand.

Bishop & Bishop (2024), Eqs. 10.9-10.10.  The network is `ToyNet`: the
four fixed kernels of Class 23 with a ReLU, then one learned-looking
3 x 3 kernel per channel summed into a single map with a ReLU, and a
class score that is the sum of that map.  The gradient of the score
with respect to every layer-1 map is written out explicitly (a ReLU
mask sent back through the transpose of each layer-2 kernel), so
alpha_k in Eq. 10.9 is an exact average and L in Eq. 10.10 an exact
weighted sum -- no autodiff, nothing approximate.

(a) The input scene.  (b) The four layer-1 maps with their alpha_k.
(c) The saliency map L, and (d) L over the image as a heat map: the
    regions the score depends on most.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import ToyNet, grad_cam, KERNELS, synthetic_image

NAME = "gradcam"


def build():
    S.use()
    X = synthetic_image(64)
    net = ToyNet()
    A1, pre2, A2, a = net.forward(X)
    L, alphas = grad_cam(net, X)
    print("class score %.2f; alphas %s" % (a, np.round(alphas, 3)))

    names = list(KERNELS.keys())
    fig, axes = plt.subplots(1, 6, figsize=(13.2, 2.9))
    axes[0].imshow(X, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("(a) input", fontsize=12, pad=8)
    for k in range(4):
        ax = axes[1 + k]
        ax.imshow(A1[k], cmap="gray")
        ax.set_title("%s\n$\\alpha_%d = %.3f$" % (names[k], k + 1, alphas[k]),
                     fontsize=11, pad=6)
    ax = axes[5]
    ax.imshow(X, cmap="gray", vmin=0, vmax=1, alpha=0.85)
    Lp = np.maximum(L, 0); Lp = Lp / Lp.max()
    ax.imshow(np.ma.masked_where(Lp < 0.04, Lp), cmap="inferno", alpha=0.8,
              vmin=0, vmax=1)
    ax.set_title("(c) saliency $\\mathbf{L}$", fontsize=12, pad=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].set_title("(b) %s\n$\\alpha_1 = %.3f$" % (names[0], alphas[0]),
                      fontsize=11, pad=6)
    fig.tight_layout(w_pad=0.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
