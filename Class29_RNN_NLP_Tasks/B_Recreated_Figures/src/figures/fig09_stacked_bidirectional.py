"""Two ways to combine RNNs as modules, in the layouts of Jurafsky &
Martin (2026), Figs. 14.10 and 14.11: (a) stacked, the outputs of one
layer are the inputs of the next; (b) bidirectional, a left-to-right
and a right-to-left RNN whose states are concatenated at every step
(Eq. 14.18)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from common import style as S
from common import draw as D

NAME = "stacked_bidirectional"


def layer(ax, y, n, color, fill, label, dx=1.0, reverse=False):
    D.band(ax, -0.5, (n - 1) * dx + 0.5, y - 0.3, y + 0.3, color, alpha=0.13)
    cs = []
    for k in range(n):
        x = k * dx
        ax.add_patch(plt.Rectangle((x - 0.17, y - 0.19), 0.34, 0.38, fc=fill, ec=color, lw=1.1, zorder=3))
        if k:
            p, q = (x - dx + 0.17, y), (x - 0.17, y)
            D.arrow(ax, q, p, color=color) if reverse else D.arrow(ax, p, q, color=color)
        cs.append((x, y))
    ax.text((n - 1) * dx + 0.62, y, label, ha="left", va="center", fontsize=11, color=color)
    return cs


def build():
    S.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    X = ["$x_1$", "$x_2$", "$x_3$", "$x_n$"]; Y = ["$y_1$", "$y_2$", "$y_3$", "$y_n$"]
    # (a) stacked
    rows = []
    for j, (col, fill) in enumerate(((D.ENC, D.ENC_F), (S.L2, "#D6E6EC"), (D.DEC, D.DEC_F))):
        rows.append(layer(a, j * 0.95, 4, col, fill, "RNN %d" % (j + 1)))
    for k in range(4):
        a.text(k, -0.8, X[k], ha="center", va="center", fontsize=11); D.arrow(a, (k, -0.62), (k, -0.21))
        for j in range(2):
            D.arrow(a, (k, j * 0.95 + 0.21), (k, (j + 1) * 0.95 - 0.21))
        D.arrow(a, (k, 2 * 0.95 + 0.21), (k, 2 * 0.95 + 0.6), color=D.DEC)
        a.text(k, 2 * 0.95 + 0.78, Y[k], ha="center", va="center", fontsize=11, color=D.DEC)
    a.text(2.5, -0.8, "$\\cdots$", ha="center", va="center", fontsize=12)
    a.set_title("(a) stacked: three layers", loc="left")
    # (b) bidirectional
    layer(b, 0.0, 4, D.ENC, D.ENC_F, "RNN 1  $\\rightarrow$")
    layer(b, 0.95, 4, S.L2, "#D6E6EC", "RNN 2  $\\leftarrow$", reverse=True)
    for k in range(4):
        b.text(k, -0.8, X[k], ha="center", va="center", fontsize=11)
        D.arrow(b, (k, -0.62), (k, -0.21))
        D.arrow(b, (k - 0.3, -0.62), (k - 0.3, 0.95 - 0.21), color=S.GREY, lw=0.7)
        # concatenated outputs
        b.add_patch(plt.Rectangle((k - 0.17, 1.72), 0.34, 0.22, fc=D.DEC_F, ec=D.DEC, lw=1.0, zorder=3))
        D.arrow(b, (k + 0.05, 0.21), (k + 0.05, 1.70), color=D.ENC, lw=0.8)
        D.arrow(b, (k + 0.12, 0.95 + 0.21), (k + 0.12, 1.70), color=S.L2, lw=0.8)
        D.arrow(b, (k, 1.96), (k, 2.28), color=D.DEC)
        b.text(k, 2.45, Y[k], ha="center", va="center", fontsize=11, color=D.DEC)
    b.text(3.62, 1.83, "$[\\mathbf{h}^f_t ; \\mathbf{h}^b_t]$", ha="left", va="center", fontsize=11, color=D.DEC)
    b.set_title("(b) bidirectional: states concatenated, Eq. 14.18", loc="left")
    for ax in (a, b):
        ax.set_xlim(-0.7, 5.0); ax.set_ylim(-1.05, 2.85); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(w_pad=0.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
