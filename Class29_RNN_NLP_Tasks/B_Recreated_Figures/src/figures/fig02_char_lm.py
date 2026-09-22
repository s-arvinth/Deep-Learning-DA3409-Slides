"""A character-level LSTM language model trained here on a short
public-domain story: (a) training and validation perplexity by step
(exp of the mean cross-entropy, Jurafsky Eq. 3.14); (b) text sampled
from the trained model at three temperatures (Section 14.3.3)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import experiment as E

NAME = "char_lm"


def build():
    S.use()
    out = E.run()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw=dict(width_ratios=[1.0, 1.35]))
    h = np.array(out["lm"]["hist"])
    a.plot(h[:, 0], h[:, 1], color=S.L1, lw=2.2, label="training")
    a.plot(h[:, 0], h[:, 2], color=S.OUTC, lw=2.2, label="validation")
    a.set_xlabel("training step"); a.set_ylabel("perplexity"); a.set_yscale("log")
    a.set_yticks([4, 6, 8, 10, 15, 20]); a.set_yticklabels(["4", "6", "8", "10", "15", "20"])
    a.legend(loc="upper right"); a.set_title("(a) perplexity while training", loc="left")
    S.square(a)
    b.axis("off"); b.set_title("(b) samples after training, prefix “Della ”", loc="left")
    y = 0.98
    for t, col in zip(E.TEMPS, (S.L2, S.L1, S.OUTC)):
        s = out["lm"]["samples"][t][:120]
        b.text(0.0, y, "temperature %.1f" % t, transform=b.transAxes, va="top", fontsize=12.5, color=col, weight="bold")
        b.text(0.0, y - 0.09, "\n".join(textwrap.wrap(s, 52)), transform=b.transAxes, va="top", fontsize=11.0,
               family="Fira Mono", color="black", linespacing=1.25)
        y -= 0.33
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
