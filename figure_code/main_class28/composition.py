"""composition — a random linear-tanh layer composed 1-5 times (Goodfellow Fig. 10.15 recomputed)

Used in: Class 28 (B_Recreated_Figures), slide(s): "The composition, drawn"
Needs:   numpy, matplotlib  (optional: sci_style)
Run:     python composition.py            -> output/composition.pdf
         python composition.py --out DIR  -> DIR/composition.pdf

Ported from Class28_LSTM_Gated_RNNs/B_Recreated_Figures/src/figures/fig07_composition.py (build()),
same code, data, parameters and seeds.  The original module's description:

    Composing many nonlinear functions, in the layout of Goodfellow et al.
    (2016), Fig. 10.15, recomputed: a random linear-tanh layer applied 0 to
    5 times to a 100-dimensional state, seen along one random line through
    the state and projected to one output.
"""
import numpy as np
import matplotlib.pyplot as plt
import common as S
from models import linear_tanh_composition

NAME = "composition"


def build():
    S.use()
    xs, outs = linear_tanh_composition(5)
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    # depth 0: the linear projection of the input line itself
    rng = np.random.default_rng(0)
    ax.plot(xs, xs / 17.0, color=S.GREY, lw=1.4, ls="-", label="0 (the input line)")
    cols = [S.L1, S.OUTC, S.L2, S.ACC, "#5B5B8F"]; styles = ["-", "-", "--", "-.", "-"]
    for k, (o, c, ls) in enumerate(zip(outs, cols, styles)):
        ax.plot(xs, o, color=c, lw=2.2 if k in (1, 4) else 1.6, ls=ls, label=str(k + 1))
    ax.set_xlabel("input coordinate (along one random direction of the state)")
    ax.set_ylabel("projection of the output")
    ax.legend(title="layers composed", loc="upper right", fontsize=11, title_fontsize=11)
    ax.set_xlim(-60, 60)
    S.square(ax); ax.set_box_aspect(0.42)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
