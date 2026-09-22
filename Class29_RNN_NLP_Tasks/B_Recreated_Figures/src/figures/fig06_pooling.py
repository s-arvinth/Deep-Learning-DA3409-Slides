"""Three read-outs for sequence classification -- the last state h_n,
the mean of the states (Jurafsky Eq. 14.15) and the elementwise max --
on a task whose evidence sits in the first three symbols: (a) the norm
of the gradient reaching the first hidden state at initialisation,
against sequence length; (b) the training loss at length 40 and 80."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import experiment as E

NAME = "pooling"
LAB = {"last": "last state $\\mathbf{h}_n$", "mean": "mean of $\\mathbf{h}_1..\\mathbf{h}_n$", "max": "elementwise max"}
COL = {"last": S.L1, "mean": S.OUTC, "max": S.ACC}


def build():
    S.use()
    out = E.run()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for ro in ("last", "mean", "max"):
        g = out["cls"][ro]["grad_init"]
        ks = sorted(g); a.plot(ks, [g[k] for k in ks], "o-", color=COL[ro], lw=2.0, ms=5, label=LAB[ro])
    a.set_yscale("log"); a.set_xlabel("sequence length $n$"); a.set_ylabel("$\\|\\partial L/\\partial \\mathbf{h}_1\\|$ at initialisation")
    a.legend(loc="lower left", fontsize=11); a.set_title("(a) how much loss reaches the first state", loc="left")
    S.square(a)
    for ro in ("last", "mean", "max"):
        for T, ls in ((40, "-"), (80, "--")):
            h = np.array(out["cls"][ro]["hist"][T]); m = np.convolve(h, np.ones(25) / 25, mode="valid")
            b.plot(np.arange(len(m)) + 12, m, color=COL[ro], lw=2.0, ls=ls, label="%s, $n = %d$" % (ro, T))
    b.set_xlabel("training step"); b.set_ylabel("training loss"); b.set_ylim(-0.02, 0.8)
    b.legend(loc="center right", fontsize=10, ncol=1); b.set_title("(b) learning at $n = 40$ (solid) and $80$ (dashed)", loc="left")
    S.square(b)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
