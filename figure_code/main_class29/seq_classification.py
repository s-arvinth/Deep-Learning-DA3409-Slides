"""seq_classification — sequence classification from the final state (Jurafsky Fig. 14.8 layout)

Used in: Class 29 (B_Recreated_Figures), slide(s): "Sequence classification, drawn"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python seq_classification.py            -> output/seq_classification.pdf
         python seq_classification.py --out DIR  -> DIR/seq_classification.pdf

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/figures/fig11_seq_classification.py (build_single()),
same code, data, parameters and seeds.  The original module's description:

    Sequence classification, in the layouts of Jurafsky & Martin (2026),
    Figs. 14.8 and 14.12.

    `seq_classification`   one RNN reads x_1..x_n; its final state h_n goes
                           through a feedforward network and a softmax to
                           the class.
    `bidir_classification` two RNNs, one in each direction; the final
                           forward state h_n^f and the final backward state
                           h_1^b are concatenated as the input to the same
                           feedforward classifier.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import common as S
import bookdraw as B

NAME = "seq_classification"


def pill(ax, x, y, label, color="black", w=0.42, h=0.7, fs=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", fc="white", ec=color, lw=1.0, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def classifier(ax, x, y0, label_in):
    """h -> FFN (trapezoid) -> softmax, stacked above (x, y0)."""
    pill(ax, x, y0, label_in, B.TGT, w=1.2, h=0.4, fs=10)
    ax.add_patch(Polygon([(x - 0.75, y0 + 0.35), (x + 0.75, y0 + 0.35), (x + 0.45, y0 + 1.2), (x - 0.45, y0 + 1.2)], closed=True, fc=B.TGT_F, ec=B.TGT, lw=1.2, zorder=3))
    ax.text(x, y0 + 0.78, "FFN", ha="center", va="center", fontsize=11.5, color=B.TGT, zorder=4)
    ax.add_patch(FancyBboxPatch((x - 0.75, y0 + 1.3), 1.5, 0.45, boxstyle="round,pad=0.02,rounding_size=0.2", fc="white", ec=B.TGT, lw=1.1, zorder=3))
    ax.text(x, y0 + 1.52, "softmax", ha="center", va="center", fontsize=10.5, color=B.TGT, zorder=4)
    B.arrow(ax, (x, y0 + 1.77), (x, y0 + 2.25), color=B.TGT, lw=1.2)
    ax.text(x, y0 + 2.45, "$\\hat{\\mathbf{y}}$", ha="center", va="center", fontsize=12, color=B.TGT)


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    xs = [0, 1.9, 3.8, 7.0]; labs = ["1", "2", "3", "n"]
    B.band(ax, -0.7, xs[-1] + 0.7, 0.0, h=1.0, color=B.SRC, label="RNN", lx=xs[-1] + 0.95)
    for k, (x, l) in enumerate(zip(xs, labs)):
        B.hidden(ax, x, 0.0, color=B.SRC, w=0.42)
        pill(ax, x, -1.3, "$\\mathbf{x}_{%s}$" % l)
        B.arrow(ax, (x, -0.94), (x, -0.23), lw=1.0)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.22, 0), (x - 0.22, 0), color=B.SRC, lw=1.3)
    ax.text(5.4, -1.3, "$\\cdots$", ha="center", va="center", fontsize=13, color=S.GREY)
    B.arrow(ax, (xs[-1], 0.23), (xs[-1], 0.85), color=B.SRC, lw=1.2)
    classifier(ax, xs[-1], 1.05, "$\\mathbf{h}_n$")
    ax.text(-0.7, 2.2, "no outputs at the earlier steps, so no loss there:\nthe one error signal is the classification's, and it is\nback-propagated through the classifier, then through\nthe RNN --- end-to-end training", ha="left", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-1.2, 10.0); ax.set_ylim(-1.9, 3.8); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "seq_classification")


if __name__ == "__main__":
    build()
