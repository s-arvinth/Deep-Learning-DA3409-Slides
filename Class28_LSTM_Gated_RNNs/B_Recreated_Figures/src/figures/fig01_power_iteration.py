"""The power iteration: (a) the theorem, one eigen-direction at a time --
the factor lambda^t for five values of lambda; (b) the measurement, the
norm ||U^t h_0|| for a random 20 x 20 matrix rescaled to spectral radius
0.9, 1.0 and 1.1."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import power_iteration

NAME = "power_iteration"
T = 60


def build():
    S.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    t = np.arange(T + 1)
    for lam, c, ls in ((0.8, S.L1, "-"), (0.9, S.L1, "--"), (1.0, S.GREY, "-"), (1.1, S.OUTC, "--"), (1.2, S.OUTC, "-")):
        a.plot(t, lam ** t, color=c, lw=2.0, ls=ls, label="$\\lambda = %.1f$" % lam)
    a.set_yscale("log"); a.set_ylim(1e-6, 1e5); a.set_xlabel("time step $t$"); a.set_ylabel("$\\lambda^t$, the factor on one component")
    a.legend(loc="upper left", fontsize=10.5, ncol=2); a.set_title("(a) one eigen-direction: $\\lambda^t$", loc="left"); S.square(a)
    for rho, c in ((0.9, S.L1), (1.0, S.GREY), (1.1, S.OUTC)):
        n = power_iteration(rho, T); b.plot(t, n, color=c, lw=2.0, label="spectral radius $%.1f$" % rho)
        print("rho %.1f: norm after %d steps = %.3g" % (rho, T, n[-1]))
    b.set_yscale("log"); b.set_xlabel("time step $t$"); b.set_ylabel("$\\|\\mathbf{U}^t \\mathbf{h}_0\\|$")
    b.legend(loc="upper left", fontsize=10.5); b.set_title("(b) a random $20 \\times 20$ matrix, measured", loc="left"); S.square(b)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
