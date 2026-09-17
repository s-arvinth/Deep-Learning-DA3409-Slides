"""Two small kernels in succession are one large kernel -- until a
nonlinearity sits between them.

(a) Two 3 x 3 kernels, a vertical-edge detector followed by a blur, and
    the single 5 x 5 kernel that equals their composition, computed by
    `compose`.  Applying the two in turn or the one gives the same map
    to machine precision; the build asserts it.
(b) With a ReLU between the two, the pair is no longer any single
    kernel: the map differs from the composed one, and the difference
    is shown.  That is the sense in which a deep stack is more than a
    big filter, and also the inductive bias Bishop describes: the big
    filter must be built from small ones.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import conv2d, compose, relu, KERNELS, synthetic_image

NAME = "composition"


def build():
    S.use()
    W1 = KERNELS["vertical edge"]; W2 = KERNELS["blur"]
    W = compose(W1, W2)
    X = synthetic_image(64)
    two = conv2d(conv2d(X, W1), W2)
    one = conv2d(X, W)
    err = np.max(np.abs(two - one)[3:-3, 3:-3])
    assert err < 1e-12
    print("linear composition: max difference %.1e (away from the border)" % err)
    with_relu = conv2d(relu(conv2d(X, W1)), W2)
    diff = with_relu - one
    print("with a ReLU between: RMS difference %.3f against map RMS %.3f"
          % (np.sqrt(np.mean(diff ** 2)), np.sqrt(np.mean(one ** 2))))

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
    for ax, K, t in zip(axes[0], (W1, W2, W), ("$\\mathbf{W}_1$: vertical edge", "$\\mathbf{W}_2$: blur",
                                                "$\\mathbf{W}_2 * \\mathbf{W}_1$: one $5 \\times 5$ kernel")):
        v = np.max(np.abs(K))
        ax.imshow(K, cmap="RdBu_r", vmin=-v, vmax=v)
        for a in range(K.shape[0]):
            for b in range(K.shape[1]):
                ax.text(b, a, "%.2f" % K[a, b], ha="center", va="center", fontsize=9,
                        color="white" if abs(K[a, b]) > 0.55 * v else "black")
        ax.set_title(t, fontsize=12, pad=8)
    v = np.max(np.abs(one))
    for ax, Mp, t in zip(axes[1], (one, with_relu, diff),
                         ("the $5 \\times 5$ kernel, one pass", "$\\mathbf{W}_1$, ReLU, then $\\mathbf{W}_2$",
                          "difference: no single kernel does this")):
        ax.imshow(Mp, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(t, fontsize=12, pad=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(w_pad=1.2, h_pad=1.6)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
