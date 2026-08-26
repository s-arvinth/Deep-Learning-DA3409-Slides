"""A normalization layer makes the layer below it scale-free.

Multiply every weight feeding into a normalized layer by a constant a.
The pre-activations are multiplied by a, and the normalizer divides by
their standard deviation, which is also multiplied by a. The output is
unchanged.

The gradient with respect to those weights is therefore scaled by 1/a:
the larger the weights become, the smaller the effective step. The
layer regulates its own learning rate.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import batch_norm

NAME = "scale_invariance"


def build():
    S.use()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, 30))
    W = rng.normal(0, np.sqrt(2.0 / 30), size=(30, 30))
    v = rng.standard_normal(30)
    a = np.geomspace(0.05, 20.0, 40)

    out_diff, gnorm, gnorm_raw = [], [], []
    for s in a:
        Ws = s * W
        A = X @ Ws.T
        Z = batch_norm(A)
        out_diff.append(np.abs(Z - batch_norm(X @ W.T)).max())
        # finite-difference gradient norm of a scalar readout
        eps = 1e-4 * s
        base = (batch_norm(X @ Ws.T) @ v).mean()
        g = np.empty(6)
        for k in range(6):
            Wp = Ws.copy(); Wp[k // 3, k % 3] += eps
            g[k] = ((batch_norm(X @ Wp.T) @ v).mean() - base) / eps
        gnorm.append(np.linalg.norm(g))
        base_r = ((X @ Ws.T) @ v).mean()
        gr = np.empty(6)
        for k in range(6):
            Wp = Ws.copy(); Wp[k // 3, k % 3] += eps
            gr[k] = (((X @ Wp.T) @ v).mean() - base_r) / eps
        gnorm_raw.append(np.linalg.norm(gr))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    ax = axes[0]
    ax.semilogx(a, np.array(out_diff), color=S.L2, lw=2.6)
    ax.set_xlabel(r"scaling $a$ applied to every weight")
    ax.set_ylabel("largest change in the layer output")
    ax.set_ylim(-0.05, 1.0)
    ax.annotate("the output does not move at all", xy=(0.09, 0.20),
                fontsize=14, color=S.L2)
    ax.set_title("(a) the forward pass is unchanged")
    S.square(ax)

    ax = axes[1]
    gn = np.array(gnorm); gr = np.array(gnorm_raw)
    i0 = int(np.argmin(np.abs(a - 1.0)))
    ax.loglog(a, gn / gn[i0], "o-", color=S.OUTC, lw=2.2, ms=5.5,
              mfc="white", mew=1.4, label="normalized layer")
    ax.loglog(a, gr / gr[i0], "s-", color=S.GREY, lw=2.0, ms=5,
              mfc="white", mew=1.4, label="unnormalized layer")
    ax.loglog(a, 1.0 / a, "--", color=S.L1, lw=1.8,
              label=r"$\propto 1/a$")
    ax.set_xlabel(r"scaling $a$ applied to every weight")
    ax.set_ylabel(r"gradient norm, relative to $a = 1$")
    ax.legend(loc="upper right", fontsize=12.5)
    ax.set_title("(b) the gradient scales as $1/a$")
    S.square(ax)

    fig.tight_layout(w_pad=1.8)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
