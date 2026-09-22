"""A left-to-right and a bidirectional RNN tagger on the same grammar:
(a) token accuracy overall and on the tokens whose tag can only be read
from the RIGHT context (an ambiguous word at the start of a sentence);
(b) the two taggers' distributions for the first word of "back the
bill" (a verb) and "back and neck ached" (a noun)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import models as M
from common import experiment as E

NAME = "bidirectional_labelling"
SENTS = (["back", "the", "bill"], ["back", "and", "neck", "ached"])


def build():
    S.use()
    out = E.run()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw=dict(width_ratios=[1.0, 1.4]))
    names = ["left-to-right", "bidirectional"]; cols = [S.L1, S.OUTC]
    x = np.arange(2)
    for k, nm in enumerate(names):
        r = out["tagger"][nm]
        a.bar(x[k] - 0.19, 100 * r["acc"], width=0.36, color=cols[k], alpha=0.45)
        a.bar(x[k] + 0.19, 100 * r["acc_right"], width=0.36, color=cols[k])
        a.text(x[k] - 0.19, 100 * r["acc"] + 1.5, "%.0f" % (100 * r["acc"]), ha="center", fontsize=11)
        a.text(x[k] + 0.19, 100 * r["acc_right"] + 1.5, "%.0f" % (100 * r["acc_right"]), ha="center", fontsize=11)
    a.bar([10], [0], color="white", alpha=0.45, label="all tokens (light)")
    a.bar([10], [0], color="black", label="needs right context (dark)")
    a.set_xticks(x); a.set_xticklabels(names); a.set_xlim(-0.6, 1.6); a.set_ylim(0, 140)
    a.set_ylabel("token accuracy (%)"); a.axhline(50, color=S.GREY, lw=0.9, ls=":")
    a.text(0.5, 44.0, "chance between\ntwo tags", ha="center", va="top", fontsize=9.5, color=S.GREY)
    a.legend(loc="upper left", fontsize=10.5, handlelength=1.0, framealpha=0.95)
    a.set_title("(a) accuracy on held-out sentences", loc="left"); S.square(a)
    # (b) distributions for the first word
    tags = ["VB", "NN"]; idx = [M.TIDX[t] for t in tags]
    groups = []
    for sent in SENTS:
        for nm in names:
            p = out["tagger"][nm]["model"].predict(sent)[0]
            groups.append((sent, nm, p[idx]))
    xg = np.arange(len(groups)); w = 0.38
    for g, (sent, nm, p) in enumerate(groups):
        b.bar(xg[g] - w / 2, p[0], width=w, color=S.L2, label="P(VB)" if g == 0 else None)
        b.bar(xg[g] + w / 2, p[1], width=w, color=S.ACC, label="P(NN)" if g == 0 else None)
        b.text(xg[g] - w / 2, p[0] + 0.02, "%.2f" % p[0], ha="center", fontsize=10)
        b.text(xg[g] + w / 2, p[1] + 0.02, "%.2f" % p[1], ha="center", fontsize=10)
    b.set_xticks(xg)
    b.set_xticklabels(["%s\n“%s …”" % (nm[:4] if nm == "left-to-right" else "bi", " ".join(s[:2]))
                       for s, nm, _ in groups], fontsize=10.5)
    b.set_xticklabels(["L→R\n“back the bill”", "bi\n“back the bill”", "L→R\n“back and neck”", "bi\n“back and neck”"], fontsize=10.5)
    b.set_ylim(0, 1.18); b.set_ylabel("probability for the first word")
    b.legend(loc="upper center", ncol=2, fontsize=11)
    b.set_title("(b) tagging “back” at the start of a sentence", loc="left")
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
