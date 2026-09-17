"""Intersection over union, and why it is the right size-free score.

(a) The metric drawn, in the layout of Bishop & Bishop (2024),
    Fig. 10.20: a predicted box (indigo) and a ground-truth box
    (maroon), the intersection shaded on the left and the union on the
    right, with the value printed.
(b) IoU against the horizontal error of the prediction, for a wide box
    and a narrow one of the same height.  The same error in pixels
    costs the narrow box far more, which is why an absolute overlap
    area would be the wrong score and a ratio is used.  Every point is
    `iou` evaluated on explicit boxes.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import style as S
from common.models import iou

NAME = "iou"

GT = (5.0, 4.0, 6.0, 4.0)
PRED = (6.4, 4.9, 6.0, 4.0)


def _rect(ax, b, **kw):
    ax.add_patch(Rectangle((b[0] - b[2] / 2, b[1] - b[3] / 2), b[2], b[3], **kw))


def build():
    S.use()
    v = iou(PRED, GT)
    print("IoU of the drawn boxes: %.3f" % v)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    for dx, what in ((0, "inter"), (11, "union")):
        g = (GT[0] + dx, GT[1], GT[2], GT[3]); p = (PRED[0] + dx, PRED[1], PRED[2], PRED[3])
        if what == "inter":
            x0 = max(g[0] - g[2] / 2, p[0] - p[2] / 2); x1 = min(g[0] + g[2] / 2, p[0] + p[2] / 2)
            y0 = max(g[1] - g[3] / 2, p[1] - p[3] / 2); y1 = min(g[1] + g[3] / 2, p[1] + p[3] / 2)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#7FBF7F", ec="none"))
            ax.text(5.7, 1.0, "area of intersection", ha="center", fontsize=12)
        else:
            _rect(ax, g, fc="#7FBF7F", ec="none"); _rect(ax, p, fc="#7FBF7F", ec="none")
            ax.text(16.7, 1.0, "area of union", ha="center", fontsize=12)
        _rect(ax, g, fc="none", ec=S.OUTC, lw=2.2)
        _rect(ax, p, fc="none", ec=S.L1, lw=2.2)
    ax.text(2.0, 7.6, "ground truth", color=S.OUTC, fontsize=11, ha="center")
    ax.text(9.4, 7.6, "prediction", color=S.L1, fontsize=11, ha="center")
    ax.text(11.0, -0.6, r"IoU $=$ intersection / union $= %.2f$" % v,
            ha="center", va="center", fontsize=12.5)
    ax.set_xlim(-0.3, 22.2); ax.set_ylim(-1.4, 8.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) the metric")

    ax = axes[1]
    dx = np.linspace(0, 6, 300)
    for w, col, lab in ((6.0, S.L1, "wide box, $6 \\times 4$"),
                        (2.0, S.OUTC, "narrow box, $2 \\times 4$")):
        vals = [iou((5.0 + d, 4.0, w, 4.0), (5.0, 4.0, w, 4.0)) for d in dx]
        ax.plot(dx, vals, color=col, lw=2.6, label=lab)
    ax.axhline(0.5, color=S.GREY, lw=1.1, ls=(0, (3, 3)))
    ax.annotate("counted correct above 0.5", xy=(3.1, 0.53), fontsize=12, color=S.GREY)
    ax.set_xlabel("horizontal error of the prediction (pixels)")
    ax.set_ylabel("IoU"); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_title("(b) the same error, two box sizes")
    ax.set_box_aspect(0.72)
    fig.tight_layout(w_pad=1.6)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
