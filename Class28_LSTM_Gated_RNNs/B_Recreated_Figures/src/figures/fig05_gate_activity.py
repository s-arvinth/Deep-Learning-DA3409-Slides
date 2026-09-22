"""What the gates do while the LSTM remembers.

The trained LSTM of the shared experiment on one length-40 sequence.
(a) The forget gate f_t of every unit, as an image: most units hold
    near one for the whole sequence -- the cell state is preserved --
    while a few open and close with the input.
(b) The mean forget gate and mean input gate against time: the input
    gate is high at the first step, when the fact arrives, and lower
    afterwards, when new symbols are noise to be ignored.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.experiment import run
from common.models import recall_batch

NAME = "gate_activity"


def build():
    S.use()
    m = run()["LSTM"]["model"]
    X, y = recall_batch(np.random.default_rng(11), 40, 1)
    hs, gates, logits = m.run(X)
    F = np.array([g["f"][0] for g in gates]).T          # (units, T)
    I = np.array([g["i"][0] for g in gates]).T
    print("prediction %d, target %d; mean forget gate %.2f, mean input gate at t=1 %.2f and after %.2f"
          % (np.argmax(logits.v), y[0], F.mean(), I[:, 0].mean(), I[:, 1:].mean()))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    im = ax.imshow(F, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("time $t$"); ax.set_ylabel("unit"); ax.set_title("(a) forget gate $\\mathbf{f}_t$, every unit")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    ax = axes[1]
    ax.plot(np.arange(1, 41), F.mean(0), color=S.L1, lw=2.4, label="mean forget gate")
    ax.plot(np.arange(1, 41), I.mean(0), color=S.OUTC, lw=2.4, label="mean input gate")
    ax.set_ylim(0, 1.05); ax.set_xlabel("time $t$"); ax.legend(fontsize=10.5, loc="center right")
    ax.set_title("(b) keep the state, admit the first input")
    S.square(ax)
    fig.tight_layout(w_pad=1.6)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
