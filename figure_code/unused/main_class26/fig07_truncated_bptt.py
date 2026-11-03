"""Truncated BPTT on the recall task at length 12: the marker at the
first step decides the label read at the last.  The marker embeddings
start identical, so telling them apart must be learned, and the only
gradient that can teach it travels the full 12 steps.  (a) the gradient
reaching the marker embeddings when the backward pass is cut to the
last k steps; (b) accuracy after training with that truncation."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import experiment as E

NAME = "truncated_bptt"


def build():
    S.use()
    out = E.run()
    r = out["trunc"]; k = np.array([x["k"] for x in r]); g = np.array([x["grad"] for x in r]); acc = np.array([x["acc"] for x in r])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    a.bar(np.arange(len(k)), g, color=[S.OUTC if v > 0 else S.GREY for v in g], width=0.6)
    for i, v in enumerate(g):
        a.text(i, v + 0.0006, "0" if v == 0 else "%.3f" % v, ha="center", fontsize=10)
    a.set_xticks(np.arange(len(k))); a.set_xticklabels([str(v) for v in k]); a.set_xlabel("truncation $k$ (steps back-propagated)")
    a.set_ylabel("$\\|\\partial L / \\partial \\mathbf{E}_{\\mathrm{marker}}\\|$ at initialisation"); a.set_ylim(0, g.max() * 1.25)
    a.set_title("(a) gradient reaching the first input, $\\tau = %d$" % E.T_TRUNC, loc="left"); S.square(a)
    b.plot(k, 100 * acc, "o-", color=S.L1, lw=2.0, ms=6)
    b.axhline(25, color=S.GREY, lw=0.9, ls=":"); b.text(1, 27.5, "chance, 4 markers", fontsize=10, color=S.GREY)
    b.axvline(E.T_TRUNC - 0.5, color=S.OUTC, lw=0.9, ls="--"); b.text(E.T_TRUNC - 0.8, 60, "$k = \\tau$", color=S.OUTC, fontsize=10.5, ha="right")
    b.set_xlabel("truncation $k$"); b.set_ylabel("accuracy after 800 steps (%)"); b.set_ylim(0, 108)
    b.set_title("(b) what the model learns", loc="left"); S.square(b)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
