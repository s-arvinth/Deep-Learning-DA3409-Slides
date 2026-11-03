"""The fast gradient sign method, on a classifier we trained.

Bishop & Bishop (2024), Eq. 10.11, run end to end: a logistic
classifier on raw pixels is trained by gradient descent to tell a disc
from a square (24 x 24, Gaussian pixel noise of standard deviation
0.25), reaching 99% on held-out images.  Each test image is then moved
by epsilon in the direction of the sign of the gradient of its negative
log-likelihood.

(a) A test image, the perturbation (which is just the sign pattern of
    the weights, scaled), and the perturbed image at epsilon = 0.05:
    the change is a fifth of the noise already in the image.
(b) Accuracy against epsilon.  The reason the tiny change works is
    linear: the activation moves by epsilon * ||w||_1, one small step
    per pixel added over 576 pixels.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import make_shapes, train_logistic, fgsm, accuracy

NAME = "fgsm"
SIDE = 24
EPS = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08])
SHOW = 0.05


def build():
    S.use()
    X, Y = make_shapes(800, SIDE)
    Xtr, Ytr, Xte, Yte = X[:500], Y[:500], X[500:], Y[500:]
    w, b = train_logistic(Xtr, Ytr)
    accs = [accuracy(fgsm(Xte, Yte, w, b, e), Yte, w, b) for e in EPS]
    print("accuracy against epsilon:", dict(zip(EPS, np.round(accs, 3))))
    print("||w||_1 = %.1f, so epsilon = %.2f moves the activation by %.2f"
          % (np.abs(w).sum(), SHOW, SHOW * np.abs(w).sum()))
    i = int(np.argmax(Yte == 1))          # a square
    x0 = Xte[i]; x1 = fgsm(Xte[i:i + 1], Yte[i:i + 1], w, b, SHOW)[0]
    p0 = 1 / (1 + np.exp(-(x0 @ w + b))); p1 = 1 / (1 + np.exp(-(x1 @ w + b)))
    print("shown image: p(square) %.3f before, %.3f after" % (p0, p1))

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.3),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for k, (img, t) in enumerate(((x0, "(a) test image\n$p(\\mathrm{square}) = %.2f$" % p0),
                                   (x1 - x0, "perturbation, $\\epsilon = %.2f$\n(sign of the gradient)" % SHOW),
                                   (x1, "perturbed\n$p(\\mathrm{square}) = %.2f$" % p1))):
        ax = axes[k]
        if k == 1:
            ax.imshow(img.reshape(SIDE, SIDE), cmap="RdBu_r", vmin=-SHOW, vmax=SHOW)
        else:
            ax.imshow(img.reshape(SIDE, SIDE), cmap="gray", vmin=-0.5, vmax=1.5)
        ax.set_title(t, fontsize=11, pad=6); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[3]
    ax.plot(EPS, accs, "o-", color=S.OUTC, lw=2.4, ms=6)
    ax.axhline(0.5, color=S.GREY, lw=1.0, ls=(0, (3, 3)))
    ax.annotate("chance", xy=(0.058, 0.53), fontsize=10.5, color=S.GREY)
    ax.set_xlabel("$\\epsilon$ (pixel scale $0$--$1$)", fontsize=12)
    ax.set_ylabel("test accuracy", fontsize=12)
    ax.set_ylim(0, 1.05); ax.set_xlim(-0.003, 0.085)
    ax.tick_params(labelsize=10.5)
    ax.set_title("(b) accuracy against $\\epsilon$", fontsize=11, pad=6)
    S.square(ax)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
