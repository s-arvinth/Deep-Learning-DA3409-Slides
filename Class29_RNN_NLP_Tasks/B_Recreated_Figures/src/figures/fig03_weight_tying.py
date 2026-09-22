"""Weight tying (Jurafsky Section 14.2.3): (a) parameters in the input
embedding and output layer against vocabulary size at d = 512, with
and without V = E^T; (b) the character model trained here both ways:
parameters and validation perplexity."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common import experiment as E

NAME = "weight_tying"
D = 512


def build():
    S.use()
    out = E.run()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    V = np.linspace(1000, 100000, 200)
    a.plot(V / 1000, 2 * V * D / 1e6, color=S.L1, lw=2.2, label="untied: $\\mathbf{E}$ and $\\mathbf{V}$, $2|V|d$")
    a.plot(V / 1000, V * D / 1e6, color=S.OUTC, lw=2.2, label="tied: $\\mathbf{V} = \\mathbf{E}^{\\top}$, $|V|d$")
    a.fill_between(V / 1000, V * D / 1e6, 2 * V * D / 1e6, color=S.OUTC, alpha=0.08)
    a.annotate("saved at $|V| = 50{,}000$:\n$%.1f$M parameters" % (50000 * D / 1e6), xy=(50, 1.5 * 50000 * D / 1e6),
               xytext=(18, 75), fontsize=11, color=S.OUTC,
               arrowprops=dict(arrowstyle="-|>", color=S.OUTC, lw=0.9))
    a.set_xlabel("vocabulary size $|V|$ (thousands)"); a.set_ylabel("parameters (millions)")
    a.legend(loc="upper left", fontsize=11.5); a.set_title("(a) embedding + output parameters, $d = %d$" % D, loc="left")
    S.square(a)
    names = ["untied", "tied"]
    npar = [out["lm_untied"]["n_params"], out["lm"]["n_params"]]
    ppl = [out["lm_untied"]["val_ppl"], out["lm"]["val_ppl"]]
    x = np.arange(2)
    b.bar(x - 0.2, np.array(npar) / 1000, width=0.38, color=S.L1, label="parameters (thousands)")
    b2 = b.twinx()
    b2.bar(x + 0.2, ppl, width=0.38, color=S.OUTC, label="validation perplexity")
    for k in range(2):
        b.text(x[k] - 0.2, npar[k] / 1000 + 0.6, "%.1fk" % (npar[k] / 1000), ha="center", fontsize=11, color=S.L1)
        b2.text(x[k] + 0.2, ppl[k] + 0.12, "%.2f" % ppl[k], ha="center", fontsize=11, color=S.OUTC)
    b.set_xticks(x); b.set_xticklabels(names); b.set_ylabel("parameters (thousands)", color=S.L1)
    b2.set_ylabel("validation perplexity", color=S.OUTC); b.set_ylim(0, 50); b2.set_ylim(0, 10)
    b.tick_params(axis="y", colors=S.L1); b2.tick_params(axis="y", colors=S.OUTC)
    b.set_title("(b) the character model, $|V| = %d$, $d = 64$" % out["text"]["vocab"], loc="left")
    S.square(b); S.square(b2)
    fig.tight_layout(w_pad=2.5)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
