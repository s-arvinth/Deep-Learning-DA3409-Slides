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

NAME = "char_lm_curve"
NAMES = ("char_lm_curve", "char_lm_samples")


def build():
    S.use()
    out = E.run()
    h = np.array(out["lm"]["hist"])
    fig, a = plt.subplots(figsize=(10.0, 4.4))
    a.plot(h[:, 0], h[:, 1], "o-", color=S.L1, lw=2.2, ms=4, label="training perplexity")
    a.plot(h[:, 0], h[:, 2], "s-", color=S.OUTC, lw=2.2, ms=4, label="validation perplexity")
    a.set_xlabel("training step"); a.set_ylabel("perplexity"); a.set_yscale("log")
    a.set_yticks([4, 6, 8, 10, 15, 20]); a.set_yticklabels(["4", "6", "8", "10", "15", "20"])
    a.axhline(h[-1, 2], color=S.OUTC, lw=0.8, ls=":")
    a.text(h[-1, 0], h[-1, 2] * 1.06, "%.2f" % h[-1, 2], ha="right", va="bottom", fontsize=11, color=S.OUTC)
    a.text(h[-1, 0], h[-1, 1] * 0.92, "%.2f" % h[-1, 1], ha="right", va="top", fontsize=11, color=S.L1)
    a.legend(loc="upper right", fontsize=12); a.set_box_aspect(0.42)
    S.save(fig, "char_lm_curve")
    fig, b = plt.subplots(figsize=(11.0, 4.2))
    b.axis("off")
    y = 0.98
    for t, col in zip(E.TEMPS, (S.L2, S.L1, S.OUTC)):
        s = out["lm"]["samples"][t][:150]
        b.text(0.0, y, "temperature %.1f" % t, transform=b.transAxes, va="top", fontsize=13, color=col, weight="bold")
        b.text(0.0, y - 0.11, "\n".join(textwrap.wrap(s, 78)), transform=b.transAxes, va="top", fontsize=12.0,
               family="Fira Mono", color="black", linespacing=1.3)
        y -= 0.34
    S.save(fig, "char_lm_samples")


if __name__ == "__main__":
    build()
