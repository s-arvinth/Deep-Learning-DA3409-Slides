#!/usr/bin/env python3
# =====================================================================
#  Figures for Class 3 — Perceptrons and the Perceptron Learning
#  Algorithm
#
#  Style: SciencePlots "science" preset via sci_style, Fira Sans type.
#  Run:   python3 make_figures.py          -> writes ../figs/*.pdf
# =====================================================================
import sci_style                      # SciencePlots 'science' preset
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)

# ---- Fira Sans everywhere, matching the slides ----------------------
for _f in glob.glob(os.path.join(HERE, "..", "fonts", "*.ttf")):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass

plt.rcParams.update({
    "text.usetex":      False,
    "font.family":      "Fira Sans",
    "mathtext.fontset": "custom",
    "mathtext.rm":      "Fira Sans",
    "mathtext.it":      "Fira Sans:italic",
    "mathtext.bf":      "Fira Sans:bold",
    "mathtext.sf":      "Fira Sans",
    "mathtext.tt":      "Fira Mono",
    "mathtext.default": "regular",
    "font.size":        11.0,
    "axes.titlesize":   11.5,
    "axes.labelsize":   11.0,
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  9.5,
    "legend.fontsize":  9.5,
    "figure.dpi":       150,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ---- shared palette -------------------------------------------------
L1   = "#341651"      # indigo   — class N / first series
L2   = "#1C7293"      # turquoise— boundary / second series
OUTC = "#800000"      # maroon   — class P / emphasis
ACC  = "#B8860B"      # amber    — fourth series
GREY = "#6E6E6E"
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[L1, L2, OUTC, ACC, GREY])

def save(fig, name):
    p = os.path.join(OUT, name + ".pdf")
    fig.savefig(p, bbox_inches="tight", dpi=400)
    plt.close(fig)
    print("wrote", os.path.basename(p))

def scatter_PN(ax, P, N, s=42):
    ax.scatter(*P.T, s=s + 8, marker="^", facecolor=OUTC, edgecolor="white",
               lw=0.7, zorder=4, label=r"$P$ (target 1)")
    ax.scatter(*N.T, s=s, marker="o", facecolor="white", edgecolor=L1,
               lw=1.4, zorder=4, label=r"$N$ (target 0)")

def boundary(ax, w, xs, **kw):
    """draw w0 + w1 x1 + w2 x2 = 0"""
    if abs(w[2]) > 1e-9:
        ax.plot(xs, -(w[0] + w[1] * xs) / w[2], **kw)


# =====================================================================
#  the running dataset: two linearly separable clouds
# =====================================================================
def make_data(seed=3, n=18):
    rng = np.random.default_rng(seed)
    P = rng.normal([1.5, 1.3], 0.45, (n, 2))
    N = rng.normal([-1.3, -1.0], 0.45, (n, 2))
    return P, N


# =====================================================================
#  1 — weights tilt the boundary; the bias shifts it
# =====================================================================
def fig_weights_bias():
    xs = np.linspace(-3, 3, 20)
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))

    # (a) same threshold, different weight vectors -> different tilt
    for w, c, lab in [((0, 1, 1), L1, r"$w=(1,1)$"),
                      ((0, 2, 0.6), L2, r"$w=(2,0.6)$"),
                      ((0, 0.5, 2), OUTC, r"$w=(0.5,2)$")]:
        boundary(ax[0], w, xs, color=c, lw=1.7, label=lab)
        ax[0].arrow(0, 0, w[1] * .38, w[2] * .38, color=c, lw=1.2,
                    head_width=0.12, length_includes_head=True, alpha=.85)
    ax[0].set(xlim=(-3, 3), ylim=(-3, 3), xlabel=r"$x_1$", ylabel=r"$x_2$",
              title="(a)  weights set the orientation")
    ax[0].legend(loc="lower right", fontsize=8.5)

    # (b) same weights, different bias -> parallel shift
    for b, c in zip([-2.0, 0.0, 2.0], [L1, L2, OUTC]):
        boundary(ax[1], (b, 1, 1), xs, color=c, lw=1.7,
                 label=rf"$w_0={b:+.0f}$")
    ax[1].set(xlim=(-3, 3), ylim=(-3, 3), xlabel=r"$x_1$", ylabel=r"$x_2$",
              title="(b)  the bias sets the offset")
    ax[1].legend(loc="lower right", fontsize=8.5)
    for a in ax:
        a.set_aspect("equal", "box")
        a.axhline(0, color=GREY, lw=0.4, ls=":")
        a.axvline(0, color=GREY, lw=0.4, ls=":")
    fig.tight_layout(w_pad=2.0)
    save(fig, "weights_bias")


# =====================================================================
#  2 — one update step, seen in input space
# =====================================================================
def fig_update():
    x = np.array([1.0, 1.6, 0.9])          # extended, a misclassified P
    w = np.array([0.2, -1.1, 0.6])         # w.x < 0  -> wrong
    w2 = w + x
    xs = np.linspace(-2.4, 2.4, 20)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))

    for a, ww, ttl in [(ax[0], w,  r"(a)  before: $w\cdot x < 0$"),
                       (ax[1], w2, r"(b)  after: $w' = w + x$")]:
        boundary(a, ww, xs, color=L2, lw=1.8)
        # shade the positive half-plane
        X, Y = np.meshgrid(np.linspace(-2.4, 2.4, 240),
                           np.linspace(-2.4, 2.4, 240))
        side = ww[0] + ww[1] * X + ww[2] * Y
        a.contourf(X, Y, (side > 0).astype(float), levels=[.5, 1.5],
                   colors=[OUTC], alpha=.09)
        a.contour(X, Y, side, levels=[0], colors=[L2], linewidths=0)
        a.arrow(0, 0, ww[1] * .7, ww[2] * .7, color=L2, lw=1.4,
                head_width=0.14, length_includes_head=True)
        a.annotate(r"$w$", (ww[1] * .7, ww[2] * .7),
                   textcoords="offset points", xytext=(6, 4),
                   color=L2, fontsize=10)
        a.scatter([x[1]], [x[2]], s=95, marker="^", facecolor=OUTC,
                  edgecolor="white", lw=0.9, zorder=5)
        a.annotate(r"$x \in P$", (x[1], x[2]), textcoords="offset points",
                   xytext=(8, -12), color=OUTC, fontsize=10)
        a.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xlabel=r"$x_1$",
              ylabel=r"$x_2$", title=ttl)
        a.set_aspect("equal", "box")
        a.axhline(0, color=GREY, lw=0.4, ls=":")
        a.axvline(0, color=GREY, lw=0.4, ls=":")
    ax[1].annotate("", xy=(w2[1] * .7, w2[2] * .7), xytext=(w[1] * .7, w[2] * .7),
                   arrowprops=dict(arrowstyle="->", color=ACC, lw=1.4))
    for a in ax:
        a.text(0, -2.22, "shaded = where the unit fires", fontsize=8.5,
               color=GREY, ha="center")
    fig.tight_layout(w_pad=2.0)
    save(fig, "update_step")


# =====================================================================
#  3 — the PLA running: boundary over successive corrections
# =====================================================================
def pla(P, N, w0, max_updates=60, rng=None):
    """returns the list of weight vectors after each correction"""
    rng = rng or np.random.default_rng(0)
    Pe = np.hstack([np.ones((len(P), 1)), P])
    Ne = np.hstack([np.ones((len(N), 1)), N])
    w = w0.astype(float).copy()
    hist = [w.copy()]
    data = [(v, +1) for v in Pe] + [(v, -1) for v in Ne]
    for _ in range(4000):
        wrong = [(v, s) for v, s in data if s * (w @ v) <= 0]
        if not wrong:
            break
        v, s = wrong[rng.integers(len(wrong))]
        w = w + s * v
        hist.append(w.copy())
        if len(hist) > max_updates:
            break
    return hist

def fig_pla_run():
    # clouds close together so the algorithm needs several corrections
    r = np.random.default_rng(4)
    P = r.normal([0.9, 0.7], 0.55, (22, 2))
    N = r.normal([-0.8, -0.6], 0.55, (22, 2))
    hist = pla(P, N, np.array([2.5, 2.2, -2.6]),
               rng=np.random.default_rng(3))
    xs = np.linspace(-3.2, 3.2, 20)
    last = len(hist) - 1
    picks = sorted(set([0, max(1, last // 3), max(2, 2 * last // 3), last]))
    while len(picks) < 4:
        picks.append(last)
    picks = picks[:4]

    fig, ax = plt.subplots(1, 4, figsize=(11.4, 3.0))
    for a, k in zip(ax, picks):
        scatter_PN(a, P, N, s=30)
        # faded trail of earlier boundaries
        for j in range(k):
            boundary(a, hist[j], xs, color=GREY, lw=0.7, alpha=.28)
        boundary(a, hist[k], xs, color=L2, lw=2.0)
        if k == 0:
            ttl = "start (random $w$)"
        elif k == len(hist) - 1:
            ttl = f"converged: {k} updates"
        else:
            ttl = f"after {k} update" + ("s" if k != 1 else "")
        a.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xlabel=r"$x_1$",
              title=ttl)
        a.set_aspect("equal", "box")
    ax[0].set_ylabel(r"$x_2$")
    ax[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "pla_run")


# =====================================================================
#  4 — the error surface for AND, three views
#      (the same problem Rojas uses in Fig. 4.5: a perceptron with
#       fixed threshold 1 learning the AND gate, error over w1,w2)
# =====================================================================
AND_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
AND_Y = np.array([0, 0, 0, 1])
AND_TH = 1.0

def and_error(W1, W2):
    """number of misclassified AND patterns for each (w1,w2)"""
    e = np.zeros_like(W1)
    for (a, b), t in zip(AND_X, AND_Y):
        fires = (W1 * a + W2 * b) >= AND_TH
        e += (fires != bool(t))
    return e

def fig_error_surface():
    g = np.linspace(-0.5, 1.5, 220)
    W1, W2 = np.meshgrid(g, g)
    E = and_error(W1, W2)

    fig = plt.figure(figsize=(11.6, 3.3))

    # (a) the surface itself, as a landscape
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    surf = ax0.plot_surface(W1, W2, E, cmap="plasma", linewidth=0,
                            antialiased=True, alpha=.95)
    surf.set_rasterized(True)
    ax0.set(xlabel=r"$w_1$", ylabel=r"$w_2$")
    ax0.set_zlabel("errors", fontsize=9.5)
    ax0.tick_params(labelsize=8)
    for a_ in (ax0.xaxis, ax0.yaxis, ax0.zaxis):
        a_.label.set_size(9.5)
    ax0.set_title("(a)  the error surface", fontsize=11, pad=-2)
    ax0.view_init(elev=32, azim=-128)
    ax0.set_box_aspect((1, 1, 0.62), zoom=1.02)

    # (b) the same surface from above, solution region outlined
    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.pcolormesh(W1, W2, E, cmap="plasma", shading="auto")
    im.set_rasterized(True)
    ax1.contour(W1, W2, E, levels=[0.5], colors="white", linewidths=1.8)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("misclassified patterns", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax1.text(1.15, 1.15, "solution\nregion", color="white", fontsize=9,
             ha="center", va="center")
    ax1.set(xlabel=r"$w_1$", ylabel=r"$w_2$",
            title="(b)  seen from above")
    ax1.set_aspect("equal", "box")

    # (c) the path the algorithm takes across that same surface
    P = AND_X[AND_Y == 1]; N = AND_X[AND_Y == 0]
    hist = pla_fixed_threshold(P, N, np.array([-0.3, 1.35]))
    H = np.array(hist)
    im2 = ax1_2 = fig.add_subplot(1, 3, 3)
    m = ax1_2.pcolormesh(W1, W2, E, cmap="plasma", shading="auto", alpha=.55)
    m.set_rasterized(True)
    ax1_2.contour(W1, W2, E, levels=[0.5], colors="white", linewidths=1.8)
    ax1_2.plot(H[:, 0], H[:, 1], color="white", lw=2.6, zorder=3)
    ax1_2.plot(H[:, 0], H[:, 1], color=L2, lw=1.5, marker="o", ms=4.5,
               mec="white", mew=0.7, zorder=4, label="PLA path")
    ax1_2.scatter([H[0, 0]], [H[0, 1]], s=75, marker="s", color=L1,
                  edgecolor="white", zorder=5, label="start")
    ax1_2.scatter([H[-1, 0]], [H[-1, 1]], s=110, marker="*", color=OUTC,
                  edgecolor="white", zorder=5, label="solution")
    ax1_2.legend(loc="lower left", fontsize=8, framealpha=.9,
                 facecolor="white", edgecolor="none")
    ax1_2.set(xlabel=r"$w_1$", ylabel=r"$w_2$",
              title="(c)  descending it")
    ax1_2.set_aspect("equal", "box")
    fig.tight_layout(w_pad=1.4)
    save(fig, "error_surface")


def pla_fixed_threshold(P, N, w0, thr=AND_TH, max_updates=40):
    """PLA on 2-D inputs with the threshold held fixed (Rojas' setup):
    only w1,w2 move, so the search lives in the plane we plot."""
    w = w0.astype(float).copy()
    hist = [w.copy()]
    data = [(v, +1) for v in P] + [(v, -1) for v in N]
    for _ in range(500):
        wrong = [(v, sgn) for v, sgn in data
                 if (sgn > 0 and w @ v < thr) or (sgn < 0 and w @ v >= thr)]
        if not wrong:
            break
        v, sgn = wrong[0]
        w = w + 0.25 * sgn * v
        hist.append(w.copy())
        if len(hist) > max_updates:
            break
    return hist


# =====================================================================
#  4b — one update, shown on the surface and in weight space
#       (our reading of Rojas, Fig. 4.11)
# =====================================================================
def fig_update_ws():
    g = np.linspace(-0.5, 1.5, 220)
    W1, W2 = np.meshgrid(g, g)
    E = and_error(W1, W2)
    w  = np.array([0.15, 0.45])          # misclassifies (1,1)
    x  = np.array([1.0, 1.0])            # the offending pattern
    w2 = w + 0.45 * x

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))

    # (a) the step drawn on the error map
    m = ax[0].pcolormesh(W1, W2, E, cmap="plasma", shading="auto", alpha=.7)
    m.set_rasterized(True)
    ax[0].contour(W1, W2, E, levels=[0.5], colors="white", linewidths=1.8)
    ax[0].annotate("", xy=w2, xytext=w,
                   arrowprops=dict(arrowstyle="-|>", color="white", lw=3.2))
    ax[0].annotate("", xy=w2, xytext=w,
                   arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8))
    ax[0].scatter(*w,  s=80, marker="s", color=L1, edgecolor="white",
                  zorder=5, label=r"$w$ (2 errors)")
    ax[0].scatter(*w2, s=110, marker="*", color=OUTC, edgecolor="white",
                  zorder=5, label=r"$w'=w+\eta x$")
    ax[0].legend(loc="lower left", fontsize=8, framealpha=.9,
                 facecolor="white", edgecolor="none")
    ax[0].set(xlabel=r"$w_1$", ylabel=r"$w_2$",
              title="(a)  the step, on the error map")
    ax[0].set_aspect("equal", "box")

    # (b) the same step in weight space: x defines the constraint plane
    gg = np.linspace(-0.5, 1.5, 300)
    A, B = np.meshgrid(gg, gg)
    ok = (A * x[0] + B * x[1]) >= AND_TH
    cf = ax[1].contourf(A, B, ok.astype(float), levels=[-.5, .5, 1.5],
                        colors=["#EFECF3", "#F3E2E2"])
    cf.set_rasterized(True)
    ax[1].contour(A, B, A * x[0] + B * x[1], levels=[AND_TH],
                  colors=[OUTC], linewidths=1.9)
    nx = x / np.linalg.norm(x)
    ax[1].annotate("", xy=w2, xytext=w,
                   arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8))
    ax[1].scatter(*w,  s=80, marker="s", color=L1, edgecolor="white",
                  zorder=5)
    ax[1].scatter(*w2, s=110, marker="*", color=OUTC, edgecolor="white",
                  zorder=5)
    ax[1].annotate(r"$w\cdot x \geq \theta$", (1.15, 1.2), color=OUTC,
                   fontsize=9.5, ha="center")
    ax[1].annotate(r"$w\cdot x < \theta$", (0.05, 0.02), color=L1,
                   fontsize=9.5, ha="left")
    ax[1].set(xlabel=r"$w_1$", ylabel=r"$w_2$",
              title=r"(b)  the constraint $x$ imposes")
    ax[1].set_aspect("equal", "box")
    fig.tight_layout(w_pad=2.0)
    save(fig, "update_ws")


# =====================================================================
#  5 — margin controls how long convergence takes
# =====================================================================
def fig_margin():
    gaps = np.array([1.6, 0.8, 0.4, 0.2, 0.1])
    counts = []
    for gap in gaps:
        trials = []
        for t in range(25):
            r = np.random.default_rng(100 + t)
            n = 25
            P = np.stack([r.uniform(gap/2, 2, n), r.uniform(-2, 2, n)], 1)
            N = np.stack([r.uniform(-2, -gap/2, n), r.uniform(-2, 2, n)], 1)
            h = pla(P, N, np.array([0.0, -1.0, 0.6]), max_updates=4000,
                    rng=np.random.default_rng(t))
            trials.append(len(h) - 1)
        counts.append(np.mean(trials))

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
    # (a) picture of the margin
    r = np.random.default_rng(11); n = 26
    gap = 1.6
    P = np.stack([r.uniform(gap/2, 2, n), r.uniform(-2, 2, n)], 1)
    N = np.stack([r.uniform(-2, -gap/2, n), r.uniform(-2, 2, n)], 1)
    scatter_PN(ax[0], P, N, s=26)
    ax[0].axvspan(-gap/2, gap/2, color=L2, alpha=.16)
    ax[0].axvline(0, color=L2, lw=1.6)
    ax[0].annotate("", xy=(-gap/2, 1.7), xytext=(gap/2, 1.7),
                   arrowprops=dict(arrowstyle="<->", color=L2, lw=1.2))
    ax[0].text(0, 1.9, "margin", color=L2, ha="center", fontsize=9)
    ax[0].set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.4), xlabel=r"$x_1$",
              ylabel=r"$x_2$", title="(a)  the margin")
    ax[0].legend(loc="lower left", fontsize=8, framealpha=.9,
                 facecolor="white", edgecolor="none")
    ax[0].set_aspect("equal", "box")

    ax[1].plot(gaps, counts, color=OUTC, marker="o", ms=6, mew=0.8,
               mec="white")
    ax[1].set(xscale="log", yscale="log", xlabel="margin width",
              ylabel="updates to converge",
              title="(b)  narrow margins cost more updates")
    ax[1].invert_xaxis()
    fig.tight_layout(w_pad=2.0)
    save(fig, "margin")


# =====================================================================
#  6 — XOR: what a single perceptron cannot do
# =====================================================================
def fig_xor():
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.scatter(*pts[y == 1].T, s=150, marker="^", facecolor=OUTC,
               edgecolor="white", lw=1.1, zorder=4, label="XOR $=1$")
    ax.scatter(*pts[y == 0].T, s=130, marker="o", facecolor="white",
               edgecolor=L1, lw=1.9, zorder=4, label="XOR $=0$")
    xs = np.linspace(-0.5, 1.5, 10)
    for th in (0.5, 1.5):
        ax.plot(xs, th - xs, color=GREY, lw=1.0, ls="--", alpha=.75)
    ax.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xlabel=r"$x_1$",
           ylabel=r"$x_2$", xticks=[0, 1], yticks=[0, 1],
           title="no line separates the classes")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal", "box")
    save(fig, "xor")



# =====================================================================
#  7 — the separating line with its two region inequalities
#      (our reading of Rojas, Fig. 3.4)
# =====================================================================
def fig_regions():
    w0, w1, w2 = -1.0, 0.9, 2.0          # 0.9 x1 + 2 x2 >= 1
    g = np.linspace(-0.6, 2.2, 400)
    X, Y = np.meshgrid(g, g)
    side = w0 + w1 * X + w2 * Y

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    cf = ax.contourf(X, Y, (side >= 0).astype(float), levels=[-.5, .5, 1.5],
                     colors=["#EFECF3", "#F3E2E2"])
    cf.set_rasterized(True)
    ax.contour(X, Y, side, levels=[0], colors=[L2], linewidths=2.0)

    # region labels, each carrying its inequality
    ax.text(0.12, 0.12, "unit stays silent\n$0.9x_1+2x_2 < 1$",
            color=L1, fontsize=10.5, ha="left", va="bottom")
    ax.text(1.98, 1.55, "unit fires\n$0.9x_1+2x_2 \\geq 1$",
            color=OUTC, fontsize=10.5, ha="right", va="top")
    # the boundary equation, written along the line
    ax.text(1.62, 0.02, "$0.9x_1+2x_2 = 1$", color=L2, fontsize=10.5,
            rotation=-24, ha="center", va="bottom")
    # weight vector, normal to the boundary
    n = np.array([w1, w2]); n = n / np.linalg.norm(n)
    foot = np.array([0.55, 0.25])
    ax.arrow(*foot, *(n * 0.55), color=L2, lw=1.6, head_width=0.08,
             length_includes_head=True, zorder=5)
    ax.annotate(r"$w=(0.9,\,2)$", foot + n * 0.55,
                textcoords="offset points", xytext=(6, 4), color=L2,
                fontsize=10)
    ax.set(xlim=(-0.6, 2.2), ylim=(-0.6, 2.2), xlabel=r"$x_1$",
           ylabel=r"$x_2$")
    ax.set_aspect("equal", "box")
    save(fig, "regions")


# =====================================================================
#  8 — input space and weight space are dual
#      (our reading of Rojas, Fig. 3.7)
# =====================================================================
def fig_duality():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # (a) input space: w is fixed, the inputs x vary
    w = np.array([0.0, 1.0, 1.4])
    g = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(g, g)
    side = w[0] + w[1] * X + w[2] * Y
    cf = ax[0].contourf(X, Y, (side >= 0).astype(float),
                        levels=[-.5, .5, 1.5],
                        colors=["#EFECF3", "#F3E2E2"])
    cf.set_rasterized(True)
    ax[0].contour(X, Y, side, levels=[0], colors=[L2], linewidths=1.9)
    n = w[1:] / np.linalg.norm(w[1:])
    ax[0].arrow(0, 0, *(n * 0.9), color=L2, lw=1.6, head_width=0.1,
                length_includes_head=True, zorder=5)
    ax[0].annotate(r"$w$", n * 0.9, textcoords="offset points",
                   xytext=(6, 4), color=L2, fontsize=11)
    ax[0].scatter([1.1], [0.9], s=95, marker="^", facecolor=OUTC,
                  edgecolor="white", lw=0.9, zorder=6)
    ax[0].annotate(r"$x$", (1.1, 0.9), textcoords="offset points",
                   xytext=(7, -12), color=OUTC, fontsize=11)
    ax[0].set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r"$x_1$", ylabel=r"$x_2$",
              title=r"(a)  input space: $w$ fixed")

    # (b) weight space: x is fixed, the weights w vary
    x = np.array([0.0, 1.1, 0.9])
    W1, W2 = np.meshgrid(g, g)
    side2 = x[1] * W1 + x[2] * W2
    cf2 = ax[1].contourf(W1, W2, (side2 >= 0).astype(float),
                         levels=[-.5, .5, 1.5],
                         colors=["#EFECF3", "#F3E2E2"])
    cf2.set_rasterized(True)
    ax[1].contour(W1, W2, side2, levels=[0], colors=[OUTC], linewidths=1.9)
    nx = x[1:] / np.linalg.norm(x[1:])
    ax[1].arrow(0, 0, *(nx * 0.9), color=OUTC, lw=1.6, head_width=0.1,
                length_includes_head=True, zorder=5)
    ax[1].annotate(r"$x$", nx * 0.9, textcoords="offset points",
                   xytext=(6, 4), color=OUTC, fontsize=11)
    ax[1].scatter([0.7], [1.0], s=85, marker="o", facecolor="white",
                  edgecolor=L2, lw=1.8, zorder=6)
    ax[1].annotate(r"$w$", (0.7, 1.0), textcoords="offset points",
                   xytext=(7, -12), color=L2, fontsize=11)
    ax[1].set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r"$w_1$", ylabel=r"$w_2$",
              title=r"(b)  weight space: $x$ fixed")
    for a in ax:
        a.set_aspect("equal", "box")
        a.axhline(0, color=GREY, lw=0.4, ls=":")
        a.axvline(0, color=GREY, lw=0.4, ls=":")
    fig.tight_layout(w_pad=6.0)

    # --- two short arrows in the gutter, tying the pictures together --
    from matplotlib.patches import FancyArrowPatch
    for y, col, lab in [(0.66, L2,   r"$w$: vector $\to$ point"),
                        (0.34, OUTC, r"$x$: point $\to$ normal")]:
        fig.add_artist(FancyArrowPatch(
            (0.455, y), (0.545, y), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=13, lw=1.5, color=col))
        fig.text(0.50, y + 0.055, lab, color=col, fontsize=9,
                 ha="center")
    save(fig, "duality")

if __name__ == "__main__":
    fig_regions()
    fig_duality()
    fig_weights_bias()
    fig_update()
    fig_pla_run()
    fig_error_surface()
    fig_update_ws()
    fig_margin()
    fig_xor()
    print("figures ->", os.path.abspath(OUT))
