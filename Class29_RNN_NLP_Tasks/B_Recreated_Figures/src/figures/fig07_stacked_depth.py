"""Stacked RNNs (Jurafsky Section 14.4.1, Goodfellow Section 10.5):
one, two and three LSTM layers at about the same number of parameters,
trained the same way on the character task: (a) validation perplexity
by step; (b) the final validation perplexity and the widths."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import experiment as E

NAME = "stacked_depth"


def build():
    S.use()
    out = E.run()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    cols = [S.L1, S.OUTC, S.ACC]
    for r, c in zip(out["depth"], cols):
        h = np.array(r["hist"])
        a.plot(h[:, 0], h[:, 2], color=c, lw=2.2, label="%d layer%s, width %d" % (r["layers"], "s" if r["layers"] > 1 else "", r["width"]))
    a.set_xlabel("training step"); a.set_ylabel("validation perplexity"); a.set_ylim(6.5, 14)
    a.legend(loc="upper right", fontsize=11); a.set_title("(a) matched parameters, $\\approx 38$k", loc="left")
    S.square(a)
    x = np.arange(len(out["depth"]))
    b.bar(x, [r["val_ppl"] for r in out["depth"]], color=cols, width=0.6)
    for k, r in enumerate(out["depth"]):
        b.text(x[k], r["val_ppl"] + 0.1, "%.2f" % r["val_ppl"], ha="center", fontsize=11.5)
        b.text(x[k], r["val_ppl"] / 2, "%.1fk\nparams" % (r["n_params"] / 1000), ha="center", va="center", fontsize=10.5, color="white")
    b.set_xticks(x); b.set_xticklabels(["%d layer%s" % (r["layers"], "s" if r["layers"] > 1 else "") for r in out["depth"]])
    b.set_ylabel("final validation perplexity"); b.set_ylim(0, 10.5)
    b.set_title("(b) after %d steps on %d characters" % (E.LM_STEPS, out["text"]["n_train"]), loc="left")
    S.square(b)
    fig.tight_layout(w_pad=2.0)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
