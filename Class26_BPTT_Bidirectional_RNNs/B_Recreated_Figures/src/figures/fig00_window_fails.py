"""Why a fixed window is not enough: Jurafsky & Martin's sentence (14.19),
"The flights the airline was canceling were full".  The verb "were"
agrees with "flights", six words back; a window of three sees only
"was canceling" and the nearer singular "airline".  The state of a
recurrent network is the alternative: a summary of everything before."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from common import style as S
from common import draw as D

NAME = "window_fails"
WORDS = ["The", "flights", "the", "airline", "was", "canceling", "were", "full"]


def build():
    S.use()
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    xs = [k * 1.45 for k in range(len(WORDS))]
    for x, w in zip(xs, WORDS):
        col = D.DEC if w in ("flights", "were") else "black"
        ax.add_patch(FancyBboxPatch((x - 0.62, -0.3), 1.24, 0.6, boxstyle="round,pad=0.03,rounding_size=0.1",
                                    fc=D.DEC_F if w in ("flights", "were") else "#F2F2F2", ec=col, lw=1.1, zorder=3))
        ax.text(x, 0, w, ha="center", va="center", fontsize=11.5, color=col, zorder=4)
    # the dependency arc
    ax.annotate("", xy=(xs[6], 0.36), xytext=(xs[1], 0.36),
                arrowprops=dict(arrowstyle="-|>", color=D.DEC, lw=1.8, connectionstyle="arc3,rad=-0.35", shrinkA=0, shrinkB=0), zorder=5)
    ax.text((xs[1] + xs[6]) / 2, 2.35, "plural agreement, six words apart", ha="center", fontsize=11.5, color=D.DEC)
    # the window, around the words it sees
    ax.add_patch(Rectangle((xs[3] - 0.72, -0.45), xs[5] - xs[3] + 1.44, 0.9, fc="none", ec=D.ENC, lw=1.8, ls=(0, (4, 2)), zorder=4))
    ax.text(xs[4], -0.72, "a window of three sees “airline was canceling” — singular — and predicts wrong",
            ha="center", va="top", fontsize=11, color=D.ENC)
    # the state, carried word by word, below
    for k in range(7):
        ax.annotate("", xy=(xs[k + 1] - 0.5, -1.45), xytext=(xs[k] + 0.5, -1.45),
                    arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=1.4, shrinkA=0, shrinkB=0), zorder=5)
        ax.add_patch(FancyBboxPatch((xs[k] - 0.45, -1.62), 0.9, 0.34, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=S.OUTC, lw=1.1, zorder=6))
        ax.text(xs[k], -1.45, "$\\mathbf{h}_{%d}$" % (k + 1), ha="center", va="center", fontsize=10, color=S.OUTC, zorder=7)
    ax.text(xs[3] + 0.7, -1.95, "a state carried word by word: “flights, plural” can still be in $\\mathbf{h}_6$ when “were” is predicted",
            ha="center", va="top", fontsize=11, color=S.OUTC)
    ax.set_xlim(-1.0, xs[-1] + 0.9); ax.set_ylim(-2.6, 2.8); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
