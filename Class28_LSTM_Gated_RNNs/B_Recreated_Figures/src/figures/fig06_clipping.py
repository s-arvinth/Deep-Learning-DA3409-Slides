"""Gradient clipping at a cliff.

The layout of Goodfellow Fig. 10.17: a recurrent net with two
parameters, h_t = w h_{t-1} + b for 50 steps, and a squared error at
the end.  Because w is multiplied by itself 50 times the loss has a
cliff at w = 1.  Gradient descent from the same start just below the
cliff, with the same step size, without clipping (a) and with norm
clipping at v = 1 (b): the unclipped step at the cliff throws the
parameters across the plane to a worse loss; the clipped steps are
bounded and settle in the ravine.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import cliff_loss, cliff_grad

NAME = "clipping"
START = np.array([0.99, 0.05]); LR = 0.02; STEPS = 40; V = 1.0


def descend(clip):
    p = START.copy(); path = [p.copy()]
    for _ in range(STEPS):
        g = cliff_grad(*p)
        if clip and np.linalg.norm(g) > V:
            g = g * V / np.linalg.norm(g)
        p = p - LR * g; path.append(p.copy())
    return np.array(path)


def build():
    S.use()
    w = np.linspace(-0.8, 1.15, 260); b = np.linspace(-0.65, 0.25, 220)
    Wg, Bg = np.meshgrid(w, b)
    Z = np.vectorize(cliff_loss)(Wg, Bg)
    Zc = np.log10(Z + 1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, clip, title in ((axes[0], False, "(a) without clipping"), (axes[1], True, "(b) with norm clipping, $v = %.0f$" % V)):
        ax.contourf(Wg, Bg, Zc, levels=24, cmap=S.SEQ, alpha=0.9)
        path = descend(clip)
        inside = (path[:, 0] > w[0]) & (path[:, 0] < w[-1]) & (path[:, 1] > b[0]) & (path[:, 1] < b[-1])
        ax.plot(path[:, 0], path[:, 1], "o-", color=S.OUTC if not clip else S.ACC, lw=1.6, ms=3.5, zorder=5)
        ax.plot(*START, "o", ms=8, mfc="white", mec="black", zorder=6)
        far = np.where(~inside)[0]
        print("%s: final (w, b) = (%.2f, %.3f), loss %.3g%s" % (title, path[-1, 0], path[-1, 1], cliff_loss(*path[-1]),
              "; left the frame at step %d" % far[0] if len(far) else ""))
        if len(far):
            ax.annotate("leaves the frame\nat step %d" % far[0], xy=(path[far[0] - 1, 0], path[far[0] - 1, 1]),
                        xytext=(-0.55, 0.12), fontsize=11, color="white",
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.0))
        ax.set_xlim(w[0], w[-1]); ax.set_ylim(b[0], b[-1])
        ax.set_xlabel("$w$"); ax.set_ylabel("$b$"); ax.set_title(title)
        ax.set_box_aspect(0.8)
    fig.tight_layout(w_pad=1.6)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
