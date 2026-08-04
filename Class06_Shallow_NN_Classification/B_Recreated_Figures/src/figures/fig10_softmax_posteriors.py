#!/usr/bin/env python3
"""Three Gaussian classes and their softmax posteriors over the plane."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import style
from common.style import L1, L2, OUTC, ACC, GREY, BRANCH
import matplotlib.pyplot as plt
import numpy as np

from common.data import three_gaussians
from common.models import softmax

NAME = "softmax_posteriors"


def build():
    style.use()
    X, y, (means, S) = three_gaussians()
    Si = np.linalg.inv(S)
    g = np.linspace(-5, 5, 400)
    G1, G2 = np.meshgrid(g, g)
    P = np.stack([G1, G2], -1)

    A = np.stack([-(0.5 * ((P - m) @ Si * (P - m)).sum(-1)) for m in means], -1)
    post = softmax(A, axis=-1)

    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.2))
    cols = [L1, L2, OUTC]
    mk = ["o", "s", "^"]
    for k in range(3):
        ax[0].scatter(X[y == k, 0], X[y == k, 1], s=22, marker=mk[k],
                      facecolor="white", edgecolor=cols[k], lw=1.1,
                      label=rf"$\mathcal{{C}}_{k+1}$")
        ax[0].contour(G1, G2, np.exp(A[..., k]), levels=[0.25, 0.6],
                      colors=[cols[k]], linewidths=1.0, alpha=.7)
    ax[0].legend(loc="lower left", fontsize=13.0, framealpha=.92,
                 facecolor="white", edgecolor="none")
    ax[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax[0].set_title(r"(a)  three classes, shared $\mathbf{\Sigma}$",
                    fontsize=15.6)

    # blend the three class colours in proportion to their posteriors,
    # then lift towards white so it prints as a tint rather than a poster
    from matplotlib.colors import to_rgb
    C = np.array([to_rgb(c) for c in cols])            # (3, 3)
    rgb = post @ C
    rgb = 1.0 - 0.68 * (1.0 - rgb)
    im = ax[1].imshow(np.clip(rgb, 0, 1), origin="lower",
                      extent=(-5, 5, -5, 5))
    im.set_rasterized(True)
    lead = post.argmax(-1)
    ax[1].contour(G1, G2, lead, levels=[0.5, 1.5], colors="black",
                  linewidths=1.4)
    ax[1].set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax[1].set_title(r"(b)  posteriors $\hat y_k(\mathbf{x})$ as colour",
                    fontsize=15.6)

    H = -(post * np.log(np.clip(post, 1e-12, 1))).sum(-1) / np.log(3)
    m = ax[2].pcolormesh(G1, G2, H, cmap=style.SEQ, shading="auto",
                         vmin=0, vmax=1)
    m.set_rasterized(True)
    cb = fig.colorbar(m, ax=ax[2], fraction=0.046, pad=0.03)
    cb.set_label("normalised entropy", fontsize=13.7)
    cb.ax.tick_params(labelsize=12.3)
    ax[2].contour(G1, G2, lead, levels=[0.5, 1.5], colors="white",
                  linewidths=1.4)
    ax[2].set(xlabel=r"$x_1$")
    ax[2].set_title(r"(c)  uncertainty peaks on the boundaries",
                    fontsize=15.6)
    for a_ in ax:
        a_.set_aspect("equal", "box")
    fig.tight_layout(w_pad=2.0)
    return style.save(fig, NAME)


if __name__ == "__main__":
    build()
