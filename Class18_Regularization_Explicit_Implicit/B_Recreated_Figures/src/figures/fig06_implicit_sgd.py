"""The two implicit penalties, and their sum, as surfaces.

The layout of Prince (2023), Fig. 9.4, for the loss of this deck's own.
Prince (2023), Eq. 9.9 says that, averaged over batch orderings,
stochastic gradient descent follows the flow of

    L~_SGD = L + (alpha/4)||grad L||^2
               + (alpha/4B) sum_b ||grad L_b - grad L||^2 ,

so the modified loss is the original plus two penalties: one on
steepness, one on disagreement between batches.  Each batch here sees
the valley slightly displaced, so the batches disagree where the valley
turns sharply and agree where it is flat.

(a) The original loss L.
(b) The gradient-descent term, penalising steepness.
(c) The stochastic term, penalising disagreement between batches.
(d) The modified loss, the sum of all three.

Both penalties vanish on the valley floor, where every gradient is
zero, so the set of minima is unchanged; what changes is the basin
around it, and therefore where a trajectory is steered.  The build
prints the maximum of each term so the two are seen in the same units.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import L, grad_L, batch_variance, valley

NAME = "implicit_sgd"

ALPHA = 0.25
B = 12
XL, YL = (-3.2, 1.4), (-0.9, 1.8)


def build():
    S.use()
    g1 = np.linspace(*XL, 230)
    g2 = np.linspace(*YL, 160)
    G1, G2 = np.meshgrid(g1, g2)
    Lg = L(G1, G2)
    Gd = np.empty_like(Lg)
    Sg = np.empty_like(Lg)
    for i in range(G1.shape[0]):
        for j in range(G1.shape[1]):
            w = np.array([G1[i, j], G2[i, j]])
            gg = grad_L(w)
            Gd[i, j] = 0.25 * ALPHA * float(gg @ gg)
            Sg[i, j] = 0.25 * ALPHA * batch_variance(w, B=B)
    Tg = Lg + Gd + Sg
    print("max over the window: L %.2f, GD term %.2f, SGD term %.2f"
          % (Lg.max(), Gd.max(), Sg.max()))
    assert np.allclose(Gd[np.abs(G2 - valley(G1)) < 0.01], 0, atol=1e-3)

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.0))
    panels = ((Lg, "(a) loss $L$"),
              (Gd, r"(b) gradient-descent term $\frac{\alpha}{4}\|\nabla L\|^2$"),
              (Sg, r"(c) stochastic term $\frac{\alpha}{4B}\sum_b\|\nabla L_b - \nabla L\|^2$"),
              (Tg, r"(d) modified loss $\tilde{L}_{\mathrm{SGD}}$"))
    for ax, (Z, title) in zip(axes.ravel(), panels):
        ax.contourf(G1, G2, Z, levels=18, cmap=S.SEQ, alpha=0.92)
        ax.contour(G1, G2, Z, levels=9, colors="white", linewidths=0.5,
                   alpha=0.45)
        ax.plot(g1, valley(g1), color="white", lw=1.8, ls=(0, (5, 3)),
                zorder=4)
        ax.set_xlim(*XL); ax.set_ylim(*YL)
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
        ax.set_title(title, fontsize=11.0, pad=4)
        ax.set_box_aspect(0.55)
        ax.tick_params(labelsize=9.5)
        ax.set_xlabel("$w_1$", fontsize=11, labelpad=1); ax.set_ylabel("$w_2$", fontsize=11, labelpad=1)
    axes[0, 0].annotate("valley of global minima", xy=(-3.05, -0.80),
                        color="white", fontsize=8.5, ha="left")
    fig.tight_layout(w_pad=0.8, h_pad=0.6)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
