"""The encoder-decoder for translation, in the layouts of Jurafsky &
Martin (2026), Figs. 14.17 and 14.18.

`seq2seq_full`      Fig. 14.17: source and target concatenated with a
                    separator, one recurrent network run over both; the
                    outputs of the source steps are ignored, the final
                    source state (teal) conditions the target steps; each
                    output word is the next input.
`seq2seq_unrolled`  Fig. 14.18: encoder and decoder as two bands,
                    h^e_n = c = h^d_0, the context also entering every
                    decoder step; softmax and outputs above the decoder.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import bookdraw as B

NAME = "seq2seq_full"
NAMES = ("seq2seq_full", "seq2seq_unrolled")
SRCW = ["the", "green", "witch", "arrived"]
TGTW = ["llegó", "la", "bruja", "verde"]


def build_full():
    S.use()
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    dx = 1.45
    xs = [k * dx for k in range(4)]; xt = [4.6 * dx / 1.45 * 1.0 + k * dx for k in range(5)]
    xt = [xs[-1] + dx + k * dx for k in range(5)]
    yw, ye, yh, ys, yo = 0.0, 1.05, 2.3, 3.5, 4.4
    B.band(ax, xs[0] - 0.6, xt[-1] + 0.6, yh, h=1.0, color=B.SRC)
    # source steps
    for k, (x, w) in enumerate(zip(xs, SRCW)):
        last = k == 3
        B.column(ax, x, yw, ye, yh, word=w, color=B.SRC, hid_fill=B.CTX_F if last else None, hid_label="$\\mathbf{h}_n$" if last else None)
        if last:
            ax.add_patch(plt.Rectangle((x - 0.25, yh - 0.25), 0.5, 0.5, fc=B.CTX_F, ec=B.CTX, lw=1.5, zorder=3)); ax.text(x, yh, "$\\mathbf{h}_n$", ha="center", va="center", fontsize=8.5, color=B.CTX, zorder=4)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.SRC, lw=1.3)
    ax.text((xs[0] + xs[-1]) / 2, ys + 0.05, "(output of the source steps is ignored)", ha="center", fontsize=9.5, color=S.GREY)
    # target steps
    tin = ["$\\langle s\\rangle$"] + TGTW
    tout = TGTW + ["$\\langle/s\\rangle$"]
    for k, (x, wi, wo) in enumerate(zip(xt, tin, tout)):
        B.column(ax, x, yw, ye, yh, ys, yo, word=wi, out=wo, color=B.TGT, rng=rng)
        prev = xs[-1] if k == 0 else xt[k - 1]
        B.arrow(ax, (prev + 0.27, yh), (x - 0.27, yh), color=B.SRC if k == 0 else B.TGT, lw=1.3)
        if k < 4:
            B.arrow(ax, (x + 0.42, yo - 0.15), (xt[k + 1] - 0.42, yw + 0.15), color=B.TGT, lw=1.0, ls=(0, (4, 2)), z=2)
    B.brace(ax, xs[0] - 0.4, xs[-1] + 0.4, yw - 0.45, "source text", color=B.SRC)
    B.brace(ax, xt[1] - 0.4, xt[-1] + 0.4, yo + 0.4, "target text", color=B.TGT, up=True)
    ax.text(xt[0], yw - 0.75, "separator", ha="center", fontsize=9.5, color=S.GREY)
    for y, lab in ((ye, "embedding\nlayer"), (yh, "hidden\nlayer(s)"), (ys, "softmax")):
        ax.text(xs[0] - 0.95, y, lab, ha="right", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(xs[0] - 2.6, xt[-1] + 0.9); ax.set_ylim(-1.5, 5.6); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "seq2seq_full")


def build_unrolled():
    S.use()
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    dx = 1.45
    xs = [k * dx for k in range(4)]; xt = [xs[-1] + 1.9 * dx + k * dx for k in range(5)]
    yw, ye, yh, ys, yo = 0.0, 1.05, 2.3, 3.5, 4.4
    B.band(ax, xs[0] - 0.6, xs[-1] + 0.6, yh, h=1.0, color=B.SRC)
    B.band(ax, xt[0] - 0.6, xt[-1] + 0.6, yh, h=1.0, color=B.TGT)
    srcw = ["$\\mathbf{x}_1$", "$\\mathbf{x}_2$", "$\\mathbf{x}_3$", "$\\mathbf{x}_n$"]
    for k, (x, w) in enumerate(zip(xs, srcw)):
        B.column(ax, x, yw, ye, yh, word=w, color=B.SRC, hid_label="$\\mathbf{h}^e_{%s}$" % ("n" if k == 3 else k + 1))
        if k:
            B.arrow(ax, (xs[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.SRC, lw=1.3)
    ax.text(xs[2] + dx / 2, yw, "$\\cdots$", ha="center", va="center", fontsize=13, color=B.SRC)
    # the context box between the bands
    cx = (xs[-1] + xt[0]) / 2
    ax.add_patch(plt.Rectangle((cx - 0.95, yh - 0.25), 1.9, 0.5, fc=B.CTX_F, ec=B.CTX, lw=1.5, zorder=3))
    ax.text(cx, yh, "$\\mathbf{h}^e_n = \\mathbf{c} = \\mathbf{h}^d_0$", ha="center", va="center", fontsize=9.5, color=B.CTX, zorder=4)
    B.arrow(ax, (xs[-1] + 0.27, yh), (cx - 0.97, yh), color=B.SRC, lw=1.3)
    B.arrow(ax, (cx + 0.97, yh), (xt[0] - 0.27, yh), color=B.CTX, lw=1.3)
    tin = ["$\\langle s\\rangle$", "$\\mathbf{y}_1$", "$\\mathbf{y}_2$", "$\\mathbf{y}_3$", "$\\mathbf{y}_{m-1}$"]
    tout = ["$\\mathbf{y}_1$", "$\\mathbf{y}_2$", "$\\mathbf{y}_3$", "$\\mathbf{y}_4$", "$\\langle/s\\rangle$"]
    for k, (x, wi, wo) in enumerate(zip(xt, tin, tout)):
        B.column(ax, x, yw, ye, yh, ys, yo, word=wi, out=wo, color=B.TGT, rng=rng, hid_label="$\\mathbf{h}^d_{%s}$" % ("m" if k == 4 else k + 1))
        if k:
            B.arrow(ax, (xt[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.TGT, lw=1.3)
            # the context into every step: from the box, under the band, up into the box
            yy = yh - 0.58 - 0.05 * k
            ax.plot([cx + 0.3 * (k - 2.5) / 2.5, cx + 0.3 * (k - 2.5) / 2.5, x - 0.12], [yh - 0.27, yy, yy], color=B.CTX, lw=0.8, zorder=2)
            B.arrow(ax, (x - 0.12, yy), (x - 0.12, yh - 0.29), color=B.CTX, lw=0.8, z=2)
        if k < 4:
            B.arrow(ax, (x + 0.42, yo - 0.15), (xt[k + 1] - 0.42, yw + 0.15), color=B.TGT, lw=1.0, ls=(0, (4, 2)), z=2)
    ax.text(xt[3] + dx / 2, yw, "$\\cdots$", ha="center", va="center", fontsize=13, color=B.TGT)
    B.brace(ax, xs[0] - 0.4, xs[-1] + 0.4, yw - 0.45, "encoder", color=B.SRC)
    B.brace(ax, xt[0] - 0.4, xt[-1] + 0.4, yw - 0.45, "decoder", color=B.TGT)
    for y, lab in ((ye, "embedding\nlayer"), (yh, "hidden\nlayer(s)"), (ys, "softmax")):
        ax.text(xs[0] - 0.95, y, lab, ha="right", va="center", fontsize=10.5, color=S.GREY)
    ax.text(cx, yh + 0.72, "the context $\\mathbf{c}$ enters every decoder step", ha="center", va="bottom", fontsize=10, color=B.CTX)
    ax.set_xlim(xs[0] - 2.6, xt[-1] + 0.9); ax.set_ylim(-1.5, 5.3); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, "seq2seq_unrolled")


def build():
    build_full(); build_unrolled()


if __name__ == "__main__":
    build()
