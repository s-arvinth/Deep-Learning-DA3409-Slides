"""Gradient clipping at a cliff, in the layout of Goodfellow et al.
(2016), Fig. 10.17: the loss surface J(w, b) of a recurrent net with
two parameters, h_t = w h_{t-1} + b for 50 steps and a squared error at
the end, drawn in three dimensions -- flat almost everywhere and
exponentially steep near w = 1, because w multiplies itself once per
step -- with gradient descent run on it from the same start, without
clipping (left) and with norm clipping (right)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import cliff_loss, cliff_grad

NAME = "clipping"
START = np.array([0.99, 0.05]); LR = 0.02; STEPS = 40; V = 1.0
ZCAP = 3.0


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
    w = np.linspace(-0.6, 1.12, 90); b = np.linspace(-0.6, 0.3, 70)
    Wg, Bg = np.meshgrid(w, b)
    Z = np.minimum(np.vectorize(cliff_loss)(Wg, Bg), ZCAP)
    fig = plt.figure(figsize=(12.0, 5.2))
    for k, (clip, title) in enumerate(((False, "without clipping"), (True, "with norm clipping, $v = 1$"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_wireframe(Wg, Bg, Z, rstride=6, cstride=6, color=S.L1, lw=0.7, alpha=0.9)
        path = descend(clip)
        zs = np.minimum(np.array([cliff_loss(*p) for p in path]), ZCAP)
        inside = (path[:, 0] > w[0]) & (path[:, 0] < w[-1]) & (path[:, 1] > b[0]) & (path[:, 1] < b[-1])
        pw = np.clip(path[:, 0], w[0], w[-1]); pb = np.clip(path[:, 1], b[0], b[-1])
        ax.plot(pw, pb, zs + 0.02, "o-", color=S.OUTC if not clip else S.ACC, lw=1.8, ms=3.5, zorder=10)
        ax.plot([START[0]], [START[1]], [zs[0] + 0.02], "o", ms=7, mfc="white", mec="black", zorder=11)
        far = np.where(~inside)[0]
        print("%s: final (w, b) = (%.2f, %.3f), loss %.3g%s" % (title, path[-1, 0], path[-1, 1], cliff_loss(*path[-1]),
              "; leaves the frame at step %d" % far[0] if len(far) else ""))
        ax.set_xlabel("$w$", labelpad=2); ax.set_ylabel("$b$", labelpad=2); ax.set_zlabel("$J(w, b)$", labelpad=2)
        ax.set_zlim(0, ZCAP); ax.view_init(elev=28, azim=-128)
        ax.set_title(title, fontsize=13)
        S.tidy3d(ax, ticks=None) if False else None
        ax.set_xticks([-0.5, 0, 0.5, 1.0]); ax.set_yticks([-0.5, 0, 0.25]); ax.set_zticks([0, 1, 2, 3])
        ax.tick_params(labelsize=9, pad=0)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    fig.tight_layout(w_pad=1.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
