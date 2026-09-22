"""Training against inference: teacher forcing.

The layout of Jurafsky & Martin (2026), Fig. 14.19, on a source the
trained fixed-context model actually reads.  (a) Inference: each
decoder input is the model's own previous output (Fig. 14.17), so an
early mistake is fed forward.  (b) Training: each decoder input is the
gold token (teacher forcing), and the loss at every step is
-log p(gold token), whose average is the sentence loss.  The per-token
losses printed are the model's real values on this example.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import draw as D
from common.experiment import run
from common import models as M

NAME = "teacher_forcing"
SRC = np.array([3, 5, 6, 2, 6, 7, 1, 6, 4])


def tok(y):
    return {M.BOS: "$\\langle s\\rangle$", M.EOS: "$\\langle/s\\rangle$"}.get(int(y), str(int(y)))


def build():
    S.use()
    m = run()["fixed"]["model"]
    tgt = list(SRC[::-1]) + [M.EOS]
    greedy, _ = m.greedy_decode(SRC[None, :]); greedy = list(greedy[0])
    # per-token losses under teacher forcing
    H = m.encode(SRC[None, :]); h = H[-1]; y_prev = M.BOS; losses = []
    for y in tgt:
        h, logits, _ = m.step(H, h, np.array([y_prev]))
        z = logits.v[0]; losses.append(-(z[y] - z.max() - np.log(np.exp(z - z.max()).sum())))
        y_prev = y
    print("greedy:", greedy, " gold:", tgt)
    print("per-token losses:", np.round(losses, 2), " mean %.3f" % np.mean(losses))

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.4))
    n = len(SRC); xs_e = [0.75 * i for i in range(n)]; x0d = xs_e[-1] + 1.3
    xs_d = [x0d + 0.75 * i for i in range(len(tgt))]
    for ax, mode in zip(axes, ("inference", "training")):
        D.band(ax, -0.4, xs_e[-1] + 0.4, -0.45, 1.35, D.ENC)
        D.band(ax, x0d - 0.4, xs_d[-1] + 0.4, -0.45, 2.25, D.DEC)
        prev = None
        for x, s_ in zip(xs_e, SRC):
            c = D.column(ax, x, 0, D.ENC, D.ENC_F, rows=("emb", "hid"), w=0.5, h=0.42)
            ax.text(x, -0.75, str(s_), ha="center", fontsize=11)
            D.arrow(ax, (x, -0.55), c["emb"]); D.arrow(ax, c["emb"], c["hid"])
            if prev is not None:
                D.arrow(ax, prev, c["hid"], color=D.ENC)
            prev = c["hid"]
        h_last = prev
        inputs = [M.BOS] + (greedy[:-1] if mode == "inference" else tgt[:-1])
        outputs = greedy if mode == "inference" else tgt
        for k, (x, yi, yo) in enumerate(zip(xs_d, inputs, outputs)):
            c = D.column(ax, x, 0, D.DEC, D.DEC_F, rows=("emb", "hid", "soft"), w=0.5, h=0.42)
            col_in = D.DEC if mode == "inference" else S.ACC
            ax.text(x, -0.75, tok(yi), ha="center", fontsize=11, color=col_in)
            D.arrow(ax, (x, -0.55), c["emb"]); D.arrow(ax, c["emb"], c["hid"]); D.arrow(ax, c["hid"], c["soft"])
            D.arrow(ax, h_last, c["hid"], color=D.ENC if k == 0 else D.DEC); h_last = c["hid"]
            wrong = mode == "inference" and yo != tgt[k]
            ax.text(x, 2.3, tok(yo), ha="center", fontsize=11, color="#C00000" if wrong else D.DEC,
                    fontweight="bold" if wrong else "normal")
            D.arrow(ax, c["soft"], (x, 2.2), color=D.DEC)
            if mode == "inference" and k < len(tgt) - 1:
                ax.annotate("", xy=(x + 0.75, -0.62), xytext=(x + 0.08, 2.32),
                            arrowprops=dict(arrowstyle="-|>", color=S.GREY, lw=0.6,
                                            connectionstyle="arc3,rad=-0.55", shrinkA=0, shrinkB=0), zorder=2)
            if mode == "training":
                ax.text(x, 2.62, "%.2f" % losses[k], ha="center", va="bottom", fontsize=9.5, color=S.GREY, rotation=90)
        if mode == "inference":
            ax.text(x0d - 0.4, 2.85, "(a) inference: each input is the model's previous output; the bold red token is where it goes wrong",
                    fontsize=11, ha="left")
        else:
            ax.text(x0d - 0.4, 3.35, "(b) training with teacher forcing: gold inputs (amber); the losses $-\\log p(\\mathrm{gold})$ above average to %.3f" % np.mean(losses),
                    fontsize=11, ha="left")
        ax.set_xlim(-0.8, xs_d[-1] + 0.6); ax.set_ylim(-1.0, 3.55 if mode == "training" else 3.05)
        ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(h_pad=0.2)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
