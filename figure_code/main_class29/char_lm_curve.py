"""char_lm_curve — training and validation perplexity by step of the tied character-level LSTM LM

Used in: Class 29 (B_Recreated_Figures), slide(s): "A language model, trained: the perplexity"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python char_lm_curve.py            -> output/char_lm_curve.pdf
         python char_lm_curve.py --out DIR  -> DIR/char_lm_curve.pdf

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/figures/fig02_char_lm.py (build()),
same code, data, parameters and seeds.  The original module's description:

    A character-level LSTM language model trained here on a short
    public-domain story: (a) training and validation perplexity by step
    (exp of the mean cross-entropy, Jurafsky Eq. 3.14); (b) text sampled
    from the trained model at three temperatures (Section 14.3.3).
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
import experiment as E

NAME = "char_lm_curve"


def build():
    S.use()
    out = E.run(stages=("lm",))
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


if __name__ == "__main__":
    build()
