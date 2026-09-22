"""The bottleneck, in the layout of Jurafsky & Martin (2026), Fig.
14.20, then condensed: (a) the encoder's last state is the only thing
the decoder ever sees of the source, every decoder step drawing on it;
(b) the same, as two blocks and one thin wire."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from common import style as S
from common import bookdraw as B

NAME = "bottleneck_book"


def build():
    S.use()
    fig, (a, b) = plt.subplots(2, 1, figsize=(12.0, 5.6), gridspec_kw=dict(height_ratios=[1.25, 1]))
    # (a) the book's layout
    xs = [0, 1.3, 2.6, 3.9]; xd = [6.6, 7.9, 9.2, 10.5]
    for k, x in enumerate(xs):
        B.hidden(a, x, 0, color=B.SRC, fill=B.SRC_F, w=0.6)
        B.arrow(a, (x, -0.9), (x, -0.32), lw=1.0)
        if k:
            B.arrow(a, (xs[k - 1] + 0.32, 0), (x - 0.32, 0), color=B.SRC, lw=1.3)
    a.add_patch(Circle((xs[-1], 0), 0.52, fc="none", ec=B.CTX, lw=1.4, ls=(0, (4, 2)), zorder=5))
    a.text(xs[-1] + 0.15, 1.0, "bottleneck", ha="center", fontsize=12, color=B.CTX)
    for k, x in enumerate(xd):
        B.hidden(a, x, 0, color=B.TGT, fill=B.TGT_F, w=0.6)
        B.arrow(a, (x, -0.9), (x, -0.32), lw=1.0); B.arrow(a, (x, 0.32), (x, 0.9), lw=1.0)
        if k:
            B.arrow(a, (xd[k - 1] + 0.32, 0), (x - 0.32, 0), color=B.TGT, lw=1.3)
            # output fed to the next input
            B.arrow(a, (xd[k - 1] + 0.3, 0.95), (x - 0.3, -0.95), color=S.GREY, lw=0.8, ls=(0, (3, 2)), z=2)
        # the bottleneck into every decoder step
        B.arrow(a, (xs[-1] + 0.5, 0.05 - 0.08 * k), (x - 0.32, 0.05), color=B.CTX, lw=0.8, ls=(0, (3, 2)), z=2)
    B.arrow(a, (xd[-1] + 0.32, 0), (xd[-1] + 0.9, 0), color=B.TGT, lw=1.3)
    a.text(xs[1] + 0.65, 1.0, "encoder", ha="center", fontsize=12, color=B.SRC); a.text(xd[1] + 0.65, 1.0, "decoder", ha="center", fontsize=12, color=B.TGT)
    a.set_xlim(-0.8, 11.8); a.set_ylim(-1.5, 1.5); a.set_aspect("equal"); a.axis("off")
    a.set_title("(a) every arrow from the source into the decoder passes through one vector", loc="left", fontsize=12)
    # (b) condensed
    b.add_patch(FancyBboxPatch((0, -0.5), 4.2, 1.0, boxstyle="round,pad=0.05,rounding_size=0.2", fc=B.SRC_F, ec=B.SRC, lw=1.4, zorder=3)); b.text(2.1, 0, "encoder", ha="center", va="center", fontsize=13, color=B.SRC)
    b.add_patch(FancyBboxPatch((7.4, -0.5), 4.2, 1.0, boxstyle="round,pad=0.05,rounding_size=0.2", fc=B.TGT_F, ec=B.TGT, lw=1.4, zorder=3)); b.text(9.5, 0, "decoder", ha="center", va="center", fontsize=13, color=B.TGT)
    b.plot([4.25, 5.3], [0, 0], color=B.SRC, lw=1.4); b.plot([5.3, 6.3], [0, 0], color=B.CTX, lw=3.2); B.arrow(b, (6.3, 0), (7.35, 0), color=B.TGT, lw=1.4)
    b.text(5.8, -0.55, "$\\mathbf{c} = \\mathbf{h}^e_n$: $d$ numbers, whatever $n$ is", ha="center", va="top", fontsize=11, color=B.CTX)
    for x in (0.7, 1.4, 2.1, 2.8, 3.5):
        B.arrow(b, (x, -1.05), (x, -0.55), lw=0.9)
    b.text(2.1, -1.35, "$n$ source tokens in", ha="center", fontsize=10.5, color=S.GREY)
    for x in (8.1, 8.8, 9.5, 10.2, 10.9):
        B.arrow(b, (x, 0.55), (x, 1.05), lw=0.9)
    b.text(9.5, 1.3, "$m$ target tokens out", ha="center", fontsize=10.5, color=S.GREY)
    b.set_xlim(-0.8, 11.8); b.set_ylim(-1.6, 1.6); b.set_aspect("equal"); b.axis("off")
    b.set_title("(b) condensed: two blocks and one thin wire", loc="left", fontsize=12)
    fig.tight_layout(h_pad=0.3)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
