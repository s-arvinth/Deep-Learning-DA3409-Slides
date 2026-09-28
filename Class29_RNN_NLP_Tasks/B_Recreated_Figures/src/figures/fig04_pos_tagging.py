"""Part-of-speech tagging as sequence labelling, in the layout of
Jurafsky & Martin (2026), Fig. 14.7: words, embeddings, the RNN band,
a softmax over the tag set at every step, and the argmax tag above.
The histograms and tags are the trained left-to-right tagger's own."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import bookdraw as B
from common import models as M
from common import experiment as E

NAME = "pos_tagging"
SENT = ["Janet", "will", "back", "the", "bill"]


def build():
    S.use()
    out = E.run()
    tg = out["tagger"]["left-to-right"]["model"]
    P = tg.predict(SENT)
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    xs = [k * 1.8 for k in range(5)]
    yw, ye, yh, ys, yt = 0.0, 1.15, 2.5, 3.8, 4.85
    B.band(ax, -0.8, xs[-1] + 0.8, yh, h=1.05, color=B.SRC, label="RNN\nlayer(s)", lx=-0.75)
    B.band(ax, -0.8, xs[-1] + 0.8, ys, h=0.8, color=B.SRC, alpha=0.10)
    for k, (x, w, p) in enumerate(zip(xs, SENT, P)):
        B.column(ax, x, yw, ye, yh, ys, word=w, color="black", bars=p / p.max())
        tag = M.TAGS[int(np.argmax(p))]
        ax.text(x, yt, tag, ha="center", va="center", fontsize=13, color=B.TGT)
        B.arrow(ax, (x, ys + 0.2), (x, yt - 0.25), color="black", lw=0.9)
        if k:
            B.arrow(ax, (xs[k - 1] + 0.27, yh), (x - 0.27, yh), color=B.SRC, lw=1.3)
    ax.text(xs[0] + 0.3, yh + 0.72, "$\\mathbf{V}\\mathbf{h}$", ha="left", va="center", fontsize=9, color=S.GREY)
    ax.text((xs[0] + xs[1]) / 2, yh + 0.2, "$\\mathbf{h}$", ha="center", fontsize=9, color=B.SRC)
    for y, lab in ((yw, "words"), (ye, "embeddings $\\mathbf{e}$"), (ys, "softmax over\ntags $\\hat{\\mathbf{y}}$"), (yt, "argmax")):
        ax.text(-1.1, y, lab, ha="right", va="center", fontsize=10.5, color=S.GREY)
    ax.set_xlim(-3.4, xs[-1] + 1.2); ax.set_ylim(-0.5, 5.4); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
