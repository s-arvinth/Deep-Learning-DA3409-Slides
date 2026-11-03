"""gm_state — the graphical model with the state h^(t) (Goodfellow Fig. 10.8 layout)

Used in: Class 29 (B_Recreated_Figures), slide(s): "The two graphs, drawn"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python gm_state.py            -> output/gm_state.pdf
         python gm_state.py --out DIR  -> DIR/gm_state.pdf

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/figures/fig13_graphical.py (build_state()),
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

NAME = "gm_state"


def node(ax, x, y, label, color, fill, dotted=False, r=0.46, fs=11.5):
    ax.add_patch(Circle((x, y), r, fc=fill if not dotted else "white", ec=color, lw=1.3, ls=(0, (2, 2)) if dotted else "-", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def arrow(ax, p, q, color, lw=1.3, rad=0.0, dashed=False):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=(0, (4, 3)) if dashed else "-",
                                                    shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=%.2f" % rad), zorder=2)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    xs = [k * 1.9 for k in range(6)]
    for k, x in enumerate(xs):
        last = k == 5
        lab = "\\ldots" if last else str(k + 1)
        node(ax, x, 1.7, "$\\mathbf{h}^{(%s)}$" % lab, S.L1, "#E4DDEB", dotted=last)
        node(ax, x, 0, "$\\mathbf{y}^{(%s)}$" % lab, S.OUTC, "#F1D6D6", dotted=last)
        arrow(ax, (x, 1.22), (x, 0.48), S.L1, dashed=last)
        if k:
            arrow(ax, (xs[k - 1] + 0.48, 1.7), (x - 0.48, 1.7), S.L1, dashed=last)
            arrow(ax, (xs[k - 1] + 0.38, 0.3), (x - 0.38, 1.4), S.OUTC, dashed=last)
    ax.set_xlim(-0.8, xs[-1] + 0.8); ax.set_ylim(-0.7, 2.5); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "gm_state")


if __name__ == "__main__":
    build()
