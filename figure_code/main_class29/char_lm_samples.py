"""char_lm_samples — text sampled from the trained character-level LSTM LM at temperatures 0.5, 1.0, 1.5

Used in: Class 29 (B_Recreated_Figures), slide(s): "A language model, trained: the samples"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python char_lm_samples.py            -> output/char_lm_samples.pdf
         python char_lm_samples.py --out DIR  -> DIR/char_lm_samples.pdf

Ported from Class29_RNN_NLP_Tasks/B_Recreated_Figures/src/figures/fig02_char_lm.py (build()),
same code, data, parameters and seeds.  The original module's description:

    A character-level LSTM language model trained here on a short
    public-domain story: (a) training and validation perplexity by step
    (exp of the mean cross-entropy, Jurafsky Eq. 3.14); (b) text sampled
    from the trained model at three temperatures (Section 14.3.3).
"""
import textwrap
import matplotlib.pyplot as plt
import common as S
import experiment as E

NAME = "char_lm_samples"


def build():
    S.use()
    out = E.run(stages=("lm",))
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
