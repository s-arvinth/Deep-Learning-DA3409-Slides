"""Gradient descent does not go where gradient flow goes.

The layout of Prince (2023), Fig. 9.3, for a loss of this deck's own.
The loss has a curved valley of global minima: every point on it has
loss zero, so L itself expresses no preference between them. Continuous
gradient flow and discrete gradient descent nevertheless end up at
different points of that valley, and the difference is systematic.

Prince (2023), Eq. 9.8 says the discrete iteration is, to leading order,
the flow of a MODIFIED loss

    L~(w) = L(w) + (alpha/4) || dL/dw ||^2 ,

which is repelled from steep places.

(a) The loss, with the continuous flow (dashed) and discrete descent
    with step alpha (markers). They separate and land apart.
(b) The extra term (alpha/4)||grad L||^2: large where the surface is
    steep, zero on the valley floor.
(c) Their sum, with the flow on L~ (solid) tracking the discrete path
    and landing with it.

The build also checks the ORDER of the claim: the plain flow's endpoint
error falls like alpha and the modified flow's like alpha^2. The two
log-log slopes are fitted and asserted before the figure is written.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib.pyplot as plt
from common import style as S
from common.models import L, grad_L, grad_L_mod, discrete_gd, flow, valley

NAME = "implicit_gd"

W0 = np.array([-1.05, 1.55])
ALPHA = 0.25
T = 20.0
ALPHAS = np.array([0.010, 0.020, 0.040, 0.080, 0.125, 0.200, 0.250])
XL, YL = (-3.2, 1.4), (-0.9, 1.8)


def _order_check():
    gp = np.empty(len(ALPHAS)); gm = np.empty(len(ALPHAS))
    for i, a in enumerate(ALPHAS):
        steps = int(round(T / a))
        end = discrete_gd(W0, a, steps)[-1]
        gp[i] = np.linalg.norm(end - flow(W0, grad_L, a * steps))
        gm[i] = np.linalg.norm(
            end - flow(W0, lambda w, a=a: grad_L_mod(w, a), a * steps))
    # the claim is asymptotic, so the order is read off the small-alpha
    # half of the range (alpha <= 0.08); the largest steps are outside it
    k = ALPHAS <= 0.08
    sp = np.polyfit(np.log(ALPHAS[k]), np.log(gp[k]), 1)[0]
    sm = np.polyfit(np.log(ALPHAS[k]), np.log(gm[k]), 1)[0]
    print("endpoint error for alpha <= 0.08: plain flow ~ alpha^%.2f, "
          "modified flow ~ alpha^%.2f" % (sp, sm))
    assert 0.85 < sp < 1.15 and sm > 1.8


def build():
    S.use()
    _order_check()

    g1 = np.linspace(*XL, 420)
    g2 = np.linspace(*YL, 320)
    G1, G2 = np.meshgrid(g1, g2)
    Lg = L(G1, G2)
    Rg = np.empty_like(Lg)
    for i in range(G1.shape[0]):
        for j in range(G1.shape[1]):
            gg = grad_L(np.array([G1[i, j], G2[i, j]]))
            Rg[i, j] = 0.25 * ALPHA * float(gg @ gg)
    Tg = Lg + Rg

    n = int(round(T / ALPHA))
    d = discrete_gd(W0, ALPHA, n)
    fp = flow(W0, grad_L, T, keep=True)
    fm = flow(W0, lambda w: grad_L_mod(w, ALPHA), T, keep=True)
    print("endpoints: flow on L (%.2f, %.2f), descent (%.2f, %.2f), "
          "flow on L~ (%.2f, %.2f)" % (*fp[-1], *d[-1], *fm[-1]))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    panels = ((Lg, "(a) loss $L$"),
              (Rg, r"(b) regularization $\frac{\alpha}{4}\|\nabla L\|^2$"),
              (Tg, r"(c) loss $+$ regularization, $\tilde{L}$"))
    for ax, (Z, title) in zip(axes, panels):
        ax.contourf(G1, G2, Z, levels=18, cmap=S.SEQ, alpha=0.92)
        ax.contour(G1, G2, Z, levels=9, colors="white", linewidths=0.5,
                   alpha=0.45)
        ax.plot(g1, valley(g1), color="white", lw=1.8, ls=(0, (5, 3)),
                zorder=4)
        ax.set_xlim(*XL); ax.set_ylim(*YL)
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
        ax.set_title(title, fontsize=15)
        ax.set_box_aspect(0.62)
    axes[0].annotate("valley of global minima", xy=(-3.05, -0.80),
                     color="white", fontsize=12, ha="left")

    # (a): plain flow against descent
    ax = axes[0]
    ax.plot(fp[:, 0], fp[:, 1], color=S.ACC, lw=2.4, ls=(0, (4, 2.5)),
            zorder=5, label="flow on $L$")
    ax.plot(d[:, 0], d[:, 1], "o-", color=S.OUTC, lw=1.4, ms=3.4, zorder=7,
            label=r"descent, $\alpha = %.2f$" % ALPHA)
    ax.plot(*W0, "o", ms=9, mfc="white", mec=S.OUTC, mew=2.0, zorder=8)
    for p, col in ((fp[-1], S.ACC), (d[-1], S.OUTC)):
        ax.plot(p[0], p[1], "*", ms=15, mfc="white", mec=col, mew=1.9, zorder=9)
    ax.legend(loc="upper right", fontsize=11.5, handlelength=1.5,
              labelspacing=0.28)

    # (c): modified flow against descent
    ax = axes[2]
    ax.plot(fm[:, 0], fm[:, 1], color=S.L2, lw=2.4, zorder=6,
            label=r"flow on $\tilde{L}$")
    ax.plot(d[:, 0], d[:, 1], "o-", color=S.OUTC, lw=1.4, ms=3.4, zorder=7,
            label=r"descent, $\alpha = %.2f$" % ALPHA)
    ax.plot(*W0, "o", ms=9, mfc="white", mec=S.OUTC, mew=2.0, zorder=8)
    for p, col in ((fm[-1], S.L2), (d[-1], S.OUTC)):
        ax.plot(p[0], p[1], "*", ms=15, mfc="white", mec=col, mew=1.9, zorder=9)
    ax.legend(loc="upper right", fontsize=11.5, handlelength=1.5,
              labelspacing=0.28)

    fig.tight_layout(w_pad=1.4)
    S.save(fig, NAME)


if __name__ == "__main__":
    build()
