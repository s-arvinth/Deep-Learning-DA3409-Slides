"""The vanishing gradient, measured.

For the three trained models of the shared experiment and their
untrained copies, the norm of the gradient of the final loss with
respect to the hidden state k steps back, ||dL_T / dh_{T-k}||, averaged
over a batch of length-40 sequences and read directly off the tape.
At initialisation all three fall with the lag.  After training the
simple RNN's gradient at the first step is 1e-30 of its value at the
last -- it has settled into a regime that forgets -- while the LSTM's
is 1e-2 and the GRU's 0.4: the gated cells have learned to carry the
gradient, which is what carrying the first symbol to the end requires.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.experiment import run

NAME = "gradient_lag"


def build():
    S.use()
    R = run()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, key, title in ((axes[0], "grad_init", "(a) at initialisation"),
                           (axes[1], "grad_trained", "(b) after training")):
        for kind, col in (("RNN", S.OUTC), ("LSTM", S.L1), ("GRU", S.L2)):
            lags, g = R[kind][key]
            ax.semilogy(lags, g / g[-1], color=col, lw=2.4, label=kind)
            print("%s %s: gradient at lag 39 relative to lag 0: %.1e" % (kind, key, g[0] / g[-1]))
        ax.set_xlabel("lag $T - k$"); ax.set_ylabel("$\\|\\partial L_T / \\partial \\mathbf{h}_{T-k}\\|$, relative")
        ax.set_title(title); ax.legend(fontsize=11)
        S.square(ax)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
