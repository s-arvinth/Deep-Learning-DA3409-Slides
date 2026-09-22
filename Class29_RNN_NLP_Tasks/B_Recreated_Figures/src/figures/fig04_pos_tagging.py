"""Part-of-speech tagging as sequence labelling, in the layout of
Jurafsky & Martin (2026), Fig. 14.7, with the tag distributions of the
left-to-right tagger trained here on the small grammar, for the book's
sentence "Janet will back the bill"."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import draw as D
from common import models as M
from common import experiment as E

NAME = "pos_tagging"
SENT = ["Janet", "will", "back", "the", "bill"]


def build():
    S.use()
    out = E.run()
    tg = out["tagger"]["left-to-right"]["model"]
    P = tg.predict(SENT)
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    xs = [0, 1.4, 2.8, 4.2, 5.6]
    D.band(ax, -0.6, 6.2, 0.45, 1.35, D.ENC)
    prev = None
    for x, w, p in zip(xs, SENT, P):
        c = D.column(ax, x, 0, D.ENC, D.ENC_F, rows=("emb", "hid", "soft"))
        ax.text(x, -0.85, w, ha="center", fontsize=13)
        D.arrow(ax, (x, -0.6), c["emb"]); D.arrow(ax, c["emb"], c["hid"]); D.arrow(ax, c["hid"], c["soft"])
        if prev is not None:
            D.arrow(ax, prev, c["hid"], color=D.ENC)
        prev = c["hid"]
        # the distribution over tags, as small bars above the softmax
        k = int(np.argmax(p))
        bx = x - 0.5 + np.arange(len(M.TAGS)) * (1.0 / len(M.TAGS))
        for j, (bxj, pj) in enumerate(zip(bx, p)):
            ax.add_patch(plt.Rectangle((bxj, 2.35), 0.09, 0.65 * pj, fc=D.DEC if j == k else S.GREY, ec="none", zorder=3))
        ax.plot([x - 0.52, x + 0.52], [2.35, 2.35], color=S.GREY, lw=0.8)
        D.arrow(ax, c["soft"], (x, 2.3), color=D.DEC)
        ax.text(x, 3.15, M.TAGS[k], ha="center", fontsize=13, color=D.DEC, weight="bold")
        ax.text(x, 3.42, "%.2f" % p[k], ha="center", fontsize=9.5, color=S.GREY)
    for k, lab in enumerate(("embeddings $\\mathbf{e}_t$", "RNN $\\mathbf{h}_t$", "softmax over tags")):
        ax.text(-0.8, k * 0.9, lab, ha="right", va="center", fontsize=11.5, color=S.GREY)
    ax.text(-0.8, 2.65, "$\\hat{\\mathbf{y}}_t$ over %d tags" % len(M.TAGS), ha="right", va="center", fontsize=11.5, color=S.GREY)
    ax.text(-0.8, 3.15, "argmax", ha="right", va="center", fontsize=11.5, color=D.DEC)
    ax.text(6.3, 2.35, "  ".join(M.TAGS), ha="left", va="bottom", fontsize=8.5, color=S.GREY)
    ax.text(6.3, 2.62, "bars: the tags in this order", ha="left", va="bottom", fontsize=9.5, color=S.GREY)
    ax.set_xlim(-3.4, 9.0); ax.set_ylim(-1.2, 3.7); ax.set_aspect("equal"); ax.axis("off")
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
