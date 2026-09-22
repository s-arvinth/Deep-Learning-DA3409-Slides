"""The theorem, verified: gradients from back-propagation through time
written by hand (Goodfellow Eqs. 10.17-10.28) against central finite
differences on a sample of entries of every parameter, and against the
tape's automatic differentiation."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import experiment as E

NAME = "bptt_check"
COL = {"E": S.GREY, "W": S.L2, "U": S.L1, "b": S.ACC, "V": S.OUTC, "c": "#999999"}
LAB = {"E": "$\\mathbf{E}$", "W": "$\\mathbf{W}$", "U": "$\\mathbf{U}$", "b": "$\\mathbf{b}$", "V": "$\\mathbf{V}$", "c": "$\\mathbf{c}$"}


def build():
    S.use()
    out = E.run()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    allv = []
    for name, p in out["check"]["pairs"].items():
        a.scatter(p["fd"], p["bptt"], s=26, color=COL[name], label=LAB[name], zorder=3, alpha=0.9)
        allv += list(p["fd"])
    lim = max(abs(v) for v in allv) * 1.15
    a.plot([-lim, lim], [-lim, lim], color=S.GREY, lw=0.9, ls=":")
    a.set_xlabel("finite difference, $(L(\\theta+\\epsilon) - L(\\theta-\\epsilon))/2\\epsilon$"); a.set_ylabel("BPTT by hand")
    a.legend(loc="upper left", fontsize=11, ncol=2); a.set_title("(a) %d entries of six parameters" % out["check"]["n_entries"], loc="left")
    S.square(a)
    names = list(out["check"]["pairs"]); x = np.arange(len(names))
    fd_err = [np.abs(out["check"]["pairs"][n]["fd"] - out["check"]["pairs"][n]["bptt"]).max() for n in names]
    tp_err = [out["check"]["pairs"][n]["tape_vs_bptt"] for n in names]
    b.bar(x - 0.2, fd_err, width=0.38, color=S.L1, label="$\\max|\\mathrm{FD} - \\mathrm{BPTT}|$")
    b.bar(x + 0.2, [max(v, 1e-18) for v in tp_err], width=0.38, color=S.OUTC, label="$\\max|\\mathrm{tape} - \\mathrm{BPTT}|$")
    b.set_yscale("log"); b.set_ylim(1e-18, 1e-6); b.set_xticks(x); b.set_xticklabels([LAB[n] for n in names])
    b.axhline(1e-16, color=S.GREY, lw=0.8, ls=":"); b.text(len(names) - 0.6, 2e-16, "double precision", ha="right", fontsize=9.5, color=S.GREY)
    b.set_ylabel("largest disagreement"); b.legend(loc="upper right", fontsize=10.5)
    b.set_title("(b) the three agree", loc="left"); S.square(b)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
