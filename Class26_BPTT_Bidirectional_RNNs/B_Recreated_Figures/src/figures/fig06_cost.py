"""The cost of the unrolled graph against the sequence length: wall-clock
seconds of the forward and backward passes (d = 32, batch 16, on the
tape), and the number of hidden states the backward pass needs stored."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import experiment as E

NAME = "cost"


def build():
    S.use()
    out = E.run()
    r = out["timing"]; T = np.array([x["T"] for x in r]); f = np.array([x["forward"] for x in r]); bk = np.array([x["backward"] for x in r])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    a.plot(T, 1e3 * f, "o-", color=S.L1, lw=2.0, label="forward")
    a.plot(T, 1e3 * bk, "s-", color=S.OUTC, lw=2.0, label="backward")
    ref = 1e3 * f[0] * T / T[0]; a.plot(T, ref, color=S.GREY, lw=0.9, ls=":", label="slope 1")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("sequence length $\\tau$"); a.set_ylabel("milliseconds")
    a.legend(loc="upper left", fontsize=11); a.set_title("(a) time of one pass, $O(\\tau)$", loc="left"); S.square(a)
    b.plot(T, [x["stored"] for x in r], "o-", color=S.L1, lw=2.0)
    b.set_xlabel("sequence length $\\tau$"); b.set_ylabel("stored states")
    b.set_title("(b) memory, $O(\\tau)$", loc="left"); S.square(b)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
