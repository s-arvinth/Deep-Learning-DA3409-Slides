"""gm_complete — the fully connected graphical model over y^(t) (Goodfellow Fig. 10.7 layout)

Used in: Class 29 (B_Recreated_Figures), slide(s): "The two graphs, drawn"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python gm_complete.py            -> output/gm_complete.pdf
         python gm_complete.py --out DIR  -> DIR/gm_complete.pdf

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/figures/fig13_graphical.py (build_complete()),
same code, data, parameters and seeds.  The original module's description:

    The RNN as a directed graphical model, in the layouts of Goodfellow
    et al. (2016), Figs. 10.7 and 10.8.

    `gm_complete`  the fully connected graph over y^(1), y^(2), ...: every
                   past observation is a parent of every later one; the
                   future, not yet observed, is dotted.
    `gm_state`     the same with the state h^(t) in the graph: every stage
                   has the same structure and can share parameters.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import common as S

NAME = "gm_complete"


def node(ax, x, y, label, color, fill, dotted=False, r=0.46, fs=11.5):
    ax.add_patch(Circle((x, y), r, fc=fill if not dotted else "white", ec=color, lw=1.3, ls=(0, (2, 2)) if dotted else "-", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def arrow(ax, p, q, color, lw=1.3, rad=0.0, dashed=False):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=(0, (4, 3)) if dashed else "-",
                                                    shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=%.2f" % rad), zorder=2)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = [k * 1.9 for k in range(6)]
    for k, x in enumerate(xs):
        last = k == 5
        node(ax, x, 0, "$\\mathbf{y}^{(%s)}$" % ("\\ldots" if last else k + 1), S.OUTC, "#F1D6D6", dotted=last)
    for i in range(5):
        for j in range(i + 1, 6):
            dashed = j == 5
            if j == i + 1:
                arrow(ax, (xs[i] + 0.48, 0), (xs[j] - 0.48, 0), S.OUTC, dashed=dashed)
            else:
                arrow(ax, (xs[i] + 0.1, 0.46), (xs[j] - 0.1, 0.46), S.OUTC, lw=1.0, rad=-0.28 - 0.08 * (j - i), dashed=dashed)
    ax.set_xlim(-0.8, xs[-1] + 0.8); ax.set_ylim(-0.9, 4.2); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "gm_complete")


if __name__ == "__main__":
    build()
