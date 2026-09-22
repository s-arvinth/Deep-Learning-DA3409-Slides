"""Which time steps the gradient comes from.  For a loss at the last
step of a length-16 sequence: (a) the norm of the gradient on each
hidden state, ||grad_{h_t} L||, from the backward recursion; (b) the
per-step terms of Eq. 10.26, the contributions of each time step to the
gradient on U, at initialisation and after training on the recall task.
Both relative to the last step."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import experiment as E

NAME = "time_contributions"


def build():
    S.use()
    out = E.run()
    c = out["contrib"]; T = c["T"]; t = np.arange(1, T + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for key, col, lab in (("dH_init", S.L1, "at initialisation"), ("dH_trained", S.OUTC, "after training")):
        v = np.array(c[key]); a.plot(t, v / v[-1], "o-", color=col, lw=2.0, ms=4.5, label=lab)
    a.set_yscale("log"); a.set_xlabel("time step $t$"); a.set_ylabel("$\\|\\nabla_{\\mathbf{h}_t} L\\| \\,/\\, \\|\\nabla_{\\mathbf{h}_\\tau} L\\|$")
    a.legend(loc="upper left", fontsize=11); a.set_title("(a) the gradient on the states, Eq. 10.20", loc="left"); S.square(a)
    for key, col, lab in (("init", S.L1, "at initialisation"), ("trained", S.OUTC, "after training")):
        v = np.array(c[key]["U"][1:]); b.plot(t[1:], v / v[-1], "s-", color=col, lw=2.0, ms=4.5, label=lab)
    b.set_yscale("log"); b.set_xlabel("time step $t$"); b.set_ylabel("$\\|\\mathrm{term}_t\\| \\,/\\, \\|\\mathrm{term}_\\tau\\|$ in $\\nabla_{\\mathbf{U}} L$")
    b.legend(loc="upper left", fontsize=11); b.set_title("(b) the per-step terms of Eq. 10.26", loc="left"); S.square(b)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
