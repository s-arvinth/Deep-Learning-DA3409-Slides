#!/usr/bin/env python3
# =====================================================================
#  Figures for Class 7 — Deep Networks
#
#  Style: SciencePlots "science" preset via sci_style.
#  Run:   python3 make_figures.py          -> writes ../figs/*.pdf
# =====================================================================
import sci_style                      # SciencePlots 'science' preset
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from matplotlib import font_manager
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------
#  Fira Sans everywhere, to match the slides.
#  Loads the .ttf files shipped in ../fonts; falls back to any system
#  installation of Fira Sans.
# ---------------------------------------------------------------------
for _f in glob.glob(os.path.join(HERE, "..", "fonts", "*.ttf")):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass

plt.rcParams.update({
    "text.usetex":      False,        # so the Fira faces are actually used
    "font.family":      "Fira Sans",
    "mathtext.fontset": "custom",
    "mathtext.rm":      "Fira Sans",
    "mathtext.it":      "Fira Sans:italic",
    "mathtext.bf":      "Fira Sans:bold",
    "mathtext.sf":      "Fira Sans",
    "mathtext.tt":      "Fira Mono",
    "mathtext.default": "regular",
    # larger, more legible type on slides
    "font.size":        11.0,
    "axes.titlesize":   11.5,
    "axes.labelsize":   11.0,
    "xtick.labelsize":  9.5,
    "ytick.labelsize":  9.5,
    "legend.fontsize":  9.5,
})

# ---------------------------------------------------------------------
#  One palette, used everywhere in the deck.
#     L1    layer 1 / network 1 / class A
#     L2    layer 2 / network 2 / activation
#     OUT_  composition / deep result / class B
#     ACC   fourth series
# ---------------------------------------------------------------------
L1   = "#341651"      # indigo
L2   = "#1C7293"      # turquoise
OUTC = "#800000"      # maroon
ACC  = "#B8860B"      # amber
GREY = "#6E6E6E"
BRANCH = [L1, L2, OUTC, ACC]          # colour per fold-branch

plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=[L1, L2, OUTC, ACC, GREY]),
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# muted tints for activation-pattern regions (no rainbow)
REGION_CMAP = ListedColormap([
    "#EFECF3", "#E1EAEF", "#F3E7E7", "#F6F1E5",
    "#E9EDE9", "#EDE6F0", "#E4EEF1", "#F8F4EC",
])

def relu(z):
    return np.maximum(z, 0.0)

def save(fig, name):
    p = os.path.join(OUT, name + ".pdf")
    fig.savefig(p, bbox_inches="tight", dpi=400)  # vector; heavy
                                                 # fills rasterised at 400dpi
    plt.close(fig)
    print("wrote", os.path.basename(p))


# =====================================================================
#  The two networks used through the lecture.
#
#  f1 :  h1 = ReLU(x+1), h2 = ReLU(x+0.5), h3 = ReLU(x-0.5)
#        y  = -1 + 4 h1 - 6 h2 + 6 h3          (a zigzag: 3 branches)
#
#  f2 :  h1'= ReLU(y+1), h2'= ReLU(y), h3'= ReLU(y-0.5)
#        y' = -1 + 2 h1' - 3.5 h2' + 3 h3'
# =====================================================================
TH1 = np.array([[1.0, 1.0], [0.5, 1.0], [-0.5, 1.0]])
PH1 = np.array([-1.0, 4.0, -6.0, 6.0])
TH2 = np.array([[1.0, 1.0], [0.0, 1.0], [-0.5, 1.0]])
PH2 = np.array([-1.0, 2.0, -3.5, 3.0])

def net(x, TH, PH):
    a = TH[:, 0][:, None] + TH[:, 1][:, None] * x[None, :]
    h = relu(a)
    return a, h, PH[0] + (PH[1:][:, None] * h).sum(axis=0)

def f1(x): return net(x, TH1, PH1)[2]
def f2(x): return net(x, TH2, PH2)[2]


# =====================================================================
#  1 — composing two networks
# =====================================================================
def fig_compose():
    x = np.linspace(-1, 1, 1500)
    y1 = f1(x)
    yy = np.linspace(y1.min(), y1.max(), 1500)

    fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.9))
    ax[0].plot(x, y1, color=L1)
    ax[0].set(xlabel=r"$x$", ylabel=r"$y=f_1(x)$",
              title=r"(a)  network 1")
    ax[1].plot(yy, f2(yy), color=L2)
    ax[1].set(xlabel=r"$y$", ylabel=r"$y'=f_2(y)$",
              title=r"(b)  network 2")
    ax[2].plot(x, f2(y1), color=OUTC)
    ax[2].set(xlabel=r"$x$", ylabel=r"$y'=f_2(f_1(x))$",
              title=r"(c)  $f_2\circ f_1$")
    for a in ax:
        a.axhline(0, color=GREY, lw=0.4, ls=":")
    fig.tight_layout(w_pad=1.8)
    save(fig, "compose")


# =====================================================================
#  2 — folding, branch by branch
# =====================================================================
def fig_fold():
    x = np.linspace(-1, 1, 2000)
    y1 = f1(x)
    d = np.diff(y1)
    turn = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0] + 1
    edges = [0] + list(turn) + [len(x) - 1]

    fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.9))
    for i in range(len(edges) - 1):
        s = slice(edges[i], edges[i + 1] + 1)
        ax[0].plot(x[s], y1[s], color=BRANCH[i % 4])
        ax[2].plot(x[s], f2(y1)[s], color=BRANCH[i % 4])
    ax[0].set(xlabel=r"$x$", ylabel=r"$y=f_1(x)$",
              title=r"(a)  $f_1$ folds the input")

    yy = np.linspace(y1.min(), y1.max(), 1500)
    ax[1].plot(yy, f2(yy), color=GREY)
    ax[1].set(xlabel=r"$y$ (folded coordinate)", ylabel=r"$y'=f_2(y)$",
              title=r"(b)  $f_2$ acts on the fold")
    ax[2].set(xlabel=r"$x$", ylabel=r"$y'$",
              title=r"(c)  one copy per branch")
    for i in turn:
        ax[0].axvline(x[i], color=GREY, lw=0.4, ls=":")
        ax[2].axvline(x[i], color=GREY, lw=0.4, ls=":")
    fig.tight_layout(w_pad=1.8)
    save(fig, "fold")


# =====================================================================
#  3 — composing in two dimensions
#      net1: 2 inputs -> 3 ReLU units -> 1 output   (convex polygons)
#      net2: that output -> 3 ReLU units -> 1 output
# =====================================================================
W2D = np.array([[ 1.4, -0.6,  0.15],      # [w1, w2, bias]
                [-0.9, -1.1,  0.35],
                [ 0.3,  1.5,  0.10]])
V2D = np.array([0.0, 1.6, -1.4, 1.2])     # [offset, weights]

def net2d(X1, X2):
    a = (W2D[:, 0][:, None, None] * X1 + W2D[:, 1][:, None, None] * X2
         + W2D[:, 2][:, None, None])
    h = relu(a)
    return a, V2D[0] + (V2D[1:][:, None, None] * h).sum(0)

def fig_compose2d():
    g = np.linspace(-2, 2, 700)
    X1, X2 = np.meshgrid(g, g)
    a, y = net2d(X1, X2)
    yprime = f2(y.ravel()).reshape(y.shape)

    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.2))

    # (a) activation pattern of the 3 units -> convex polygonal regions
    pattern = ((a[0] > 0).astype(int) * 4 + (a[1] > 0).astype(int) * 2
               + (a[2] > 0).astype(int))
    _cf0 = ax[0].contourf(X1, X2, pattern, levels=np.arange(-.5, 8.5, 1),
                          cmap=REGION_CMAP)
    _cf0.set_rasterized(True)          # keep the fill light; text stays vector
    for k in range(3):
        w1, w2, b = W2D[k]
        if abs(w2) > 1e-9:
            ax[0].plot(g, -(w1 * g + b) / w2, color=BRANCH[k], lw=1.2,
                       label=rf"$h_{k+1}=0$")
    ax[0].legend(loc="lower left", fontsize=8.5)
    ax[0].set(xlim=(-2, 2), ylim=(-2, 2), xlabel=r"$x_1$", ylabel=r"$x_2$",
              title=r"(a)  net 1: convex regions")

    # (b) the scalar output of network 1 — piecewise planar
    c = ax[1].contourf(X1, X2, y, levels=24, cmap="viridis")
    c.set_rasterized(True)
    ax[1].contour(X1, X2, pattern, levels=np.arange(-.5, 8.5, 1),
                  colors="w", linewidths=0.6, alpha=.8)
    ax[1].set(xlabel=r"$x_1$", ylabel=r"$x_2$",
              title=r"(b)  net 1 output $y$")
    fig.colorbar(c, ax=ax[1], fraction=0.046, pad=0.03)

    # (c) after the second network — each region is cut again
    c2 = ax[2].contourf(X1, X2, yprime, levels=24, cmap="plasma")
    c2.set_rasterized(True)
    ax[2].contour(X1, X2, pattern, levels=np.arange(-.5, 8.5, 1),
                  colors="w", linewidths=0.6, alpha=.8)
    ax[2].set(xlabel=r"$x_1$", ylabel=r"$x_2$",
              title=r"(c)  after net 2: $f_2(y)$")
    fig.colorbar(c2, ax=ax[2], fraction=0.046, pad=0.03)
    for a_ in ax:
        a_.set_aspect("equal", "box")
    fig.tight_layout(w_pad=1.4)
    save(fig, "compose2d")


# =====================================================================
#  4 — the two-layer computation, stage by stage
# =====================================================================
def fig_buildup():
    x = np.linspace(-1, 1, 1500)
    a1, h1, y1 = net(x, TH1, PH1)

    fig, ax = plt.subplots(2, 3, figsize=(10.2, 5.0))
    lab = [r"$h_1=\mathrm{ReLU}(x+1)$",
           r"$h_2=\mathrm{ReLU}(x+0.5)$",
           r"$h_3=\mathrm{ReLU}(x-0.5)$"]
    for k in range(3):
        ax[0, k].plot(x, a1[k], color=GREY, lw=0.8, ls="--",
                      label="pre-activation")
        ax[0, k].plot(x, h1[k], color=L1, label="after ReLU")
        ax[0, k].set(xlabel=r"$x$", ylabel=rf"$h_{k+1}$", title=lab[k])
        ax[0, k].axhline(0, color=GREY, lw=0.4, ls=":")
    ax[0, 0].legend(loc="upper left")

    for k in range(3):
        ax[1, 0].plot(x, PH1[k + 1] * h1[k], color=BRANCH[k],
                      label=rf"$\phi_{k+1}h_{k+1}$")
    ax[1, 0].set(xlabel=r"$x$", ylabel=r"$\phi_d h_d$",
                 title=r"(d)  weighted units")
    ax[1, 0].legend()
    ax[1, 0].axhline(0, color=GREY, lw=0.4, ls=":")

    ax[1, 1].plot(x, y1, color=L1)
    ax[1, 1].set(xlabel=r"$x$", ylabel=r"$y$",
                 title=r"(e)  sum $+$ offset $=f_1(x)$")
    ax[1, 1].axhline(0, color=GREY, lw=0.4, ls=":")

    ax[1, 2].plot(x, f2(y1), color=OUTC)
    ax[1, 2].set(xlabel=r"$x$", ylabel=r"$y'$",
                 title=r"(f)  through layer 2")
    ax[1, 2].axhline(0, color=GREY, lw=0.4, ls=":")

    fig.tight_layout(w_pad=1.5, h_pad=1.7)
    save(fig, "buildup")


# =====================================================================
#  5 — linear regions per parameter
#      (a) regions vs parameter count, shallow and several depths
#      (b) FIXED parameter budget: for each K take the widest D that
#          fits the budget, then plot the regions that buys.
# =====================================================================
def p_shallow(D):        return 3 * D + 1
def p_deep(D, K):        return 2 * D + (K - 1) * D * (D + 1) + D + 1
def r_shallow(D):        return D + 1.0
def r_deep(D, K):        return (D + 1.0) ** K

def fig_regions():
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.4))

    pal = plt.cm.plasma(np.linspace(0.05, 0.75, 4))
    D = np.arange(2, 61)
    ax[0].plot(p_shallow(D), r_shallow(D), color=pal[0], marker="o",
               ms=4.0, markevery=7, mew=0.6, mec="w",
               label=r"shallow ($K=1$)")
    for i, (K, m) in enumerate([(2, "s"), (3, "^"), (5, "D")], start=1):
        Dk = np.arange(2, 31)
        ax[0].plot(p_deep(Dk, K), r_deep(Dk, K), color=pal[i], marker=m,
                   ms=4.0, markevery=5, mew=0.6, mec="w",
                   label=rf"deep, $K={K}$")
    ax[0].set(xscale="log", yscale="log",
              xlabel="number of parameters",
              ylabel="max linear regions",
              title="(a)  regions bought per parameter")
    ax[0].legend(loc="upper left")

    # (b) genuinely fixed budgets
    Ks = np.arange(1, 11)
    pal2 = plt.cm.plasma(np.linspace(0.1, 0.7, 3))
    for (P, m), c in zip([(200, "s"), (1000, "^"), (5000, "D")], pal2):
        best = []
        for K in Ks:
            Dfit = [d for d in range(1, 400)
                    if (p_shallow(d) if K == 1 else p_deep(d, K)) <= P]
            if not Dfit:
                best.append(np.nan); continue
            d = max(Dfit)
            best.append(r_shallow(d) if K == 1 else r_deep(d, K))
        ax[1].plot(Ks, best, color=c, marker=m, ms=4.6, mew=0.6,
                   mec="w", label=rf"budget $P={P}$")
    ax[1].set(yscale="log", xlabel=r"number of layers $K$",
              ylabel="max linear regions",
              title="(b)  same budget, split over more layers")
    ax[1].legend(loc="lower right")

    fig.tight_layout(w_pad=2.0)
    save(fig, "regions")


# =====================================================================
#  6 — representation learning on two half-moons
# =====================================================================
def fig_representation():
    def moons(n, noise, rng):
        t = np.linspace(0, np.pi, n)
        A = np.stack([np.cos(t), np.sin(t)], 1)
        B = np.stack([1 - np.cos(t), 0.4 - np.sin(t)], 1)
        X = np.vstack([A, B]) + noise * rng.normal(size=(2 * n, 2))
        return X, np.hstack([np.zeros(n), np.ones(n)])

    def train(seed, steps=6000, lr=0.5):
        rng = np.random.default_rng(seed)
        X, Y = moons(200, 0.10, rng)
        Xs = (X - X.mean(0)) / X.std(0)
        H1, H2 = 24, 2
        W1 = rng.normal(0, 1.0, (2, H1));            b1 = np.zeros(H1)
        W2 = rng.normal(0, np.sqrt(2 / H1), (H1, H2)); b2 = np.full(H2, 0.1)
        W3 = rng.normal(0, np.sqrt(2 / H2), (H2, 1)); b3 = np.zeros(1)
        hist = []
        for t in range(steps):
            Z1 = relu(Xs @ W1 + b1)
            Z2 = relu(Z1 @ W2 + b2)
            logit = (Z2 @ W3 + b3).ravel()
            p = 1 / (1 + np.exp(-logit))
            loss = -np.mean(Y * np.log(p + 1e-9) + (1 - Y) * np.log(1 - p + 1e-9))
            if t % 50 == 0: hist.append(loss)
            d = (p - Y) / len(Y)
            gW3 = Z2.T @ d[:, None]; gb3 = d.sum(keepdims=True)
            dZ2 = d[:, None] @ W3.T * (Z2 > 0)
            gW2 = Z1.T @ dZ2;        gb2 = dZ2.sum(0)
            dZ1 = dZ2 @ W2.T * (Z1 > 0)
            gW1 = Xs.T @ dZ1;        gb1 = dZ1.sum(0)
            for arr, gr in ((W3, gW3), (b3, gb3), (W2, gW2),
                            (b2, gb2), (W1, gW1), (b1, gb1)):
                arr -= lr * gr
        Z1 = relu(Xs @ W1 + b1); Z2 = relu(Z1 @ W2 + b2)
        logit = (Z2 @ W3 + b3).ravel()
        acc = ((logit > 0).astype(float) == Y).mean()
        return X, Y, Z2, W3, b3, acc, hist, (Z2.std(0) > 1e-3).all()

    for seed in range(25):
        X, Y, Z2, W3, b3, acc, hist, ok = train(seed)
        if ok and acc > 0.97:
            break

    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.2))
    ax[0].scatter(*X[Y == 0].T, s=22, marker="o", facecolor=L1,
                  edgecolor="w", lw=0.5, alpha=.9, label="class 0")
    ax[0].scatter(*X[Y == 1].T, s=26, marker="^", facecolor=OUTC,
                  edgecolor="w", lw=0.5, alpha=.9, label="class 1")
    ax[0].legend(loc="upper right")
    ax[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$",
              title=r"(a)  input space")
    ax[0].set_aspect("equal", "box")

    ax[1].scatter(*Z2[Y == 0].T, s=22, marker="o", facecolor=L1,
                  edgecolor="w", lw=0.5, alpha=.9)
    ax[1].scatter(*Z2[Y == 1].T, s=26, marker="^", facecolor=OUTC,
                  edgecolor="w", lw=0.5, alpha=.9)
    zx = np.linspace(Z2[:, 0].min(), Z2[:, 0].max(), 50)
    if abs(W3[1, 0]) > 1e-9:
        ax[1].plot(zx, -(W3[0, 0] * zx + b3[0]) / W3[1, 0],
                   color="k", lw=1.0, ls="--", label=r"$\mathbf{w}^\top\mathbf{z}+b=0$")
        ax[1].legend(loc="upper right")
    ax[1].set(xlabel=r"$z_1$", ylabel=r"$z_2$",
              title=rf"(b)  last hidden layer ({acc*100:.0f}\% acc.)")

    ax[2].plot(np.arange(len(hist)) * 50, hist, color=L2)
    ax[2].set(xlabel="gradient-descent step", ylabel="cross-entropy loss",
              title="(c)  training loss")
    fig.tight_layout(w_pad=1.7)
    save(fig, "representation")


# =====================================================================
#  7 — a hierarchy of features
# =====================================================================
def fig_hierarchy():
    x = np.linspace(0, 1, 2000)
    bump = lambda c, w: relu(1 - np.abs(x - c) / w)

    fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.8))
    for c, col in zip([0.2, 0.5, 0.8], BRANCH):
        ax[0].plot(x, bump(c, 0.12), color=col)
    ax[0].set(xlabel=r"$x$", ylabel="level-1 units",
              title=r"(a)  parts")

    motif = bump(0.2, .12) + 0.8 * bump(0.5, .12) + bump(0.8, .12)
    ax[1].plot(x, motif, color=GREY)
    ax[1].set(xlabel=r"$x$", ylabel="level-2 unit",
              title=r"(b)  a motif")

    ax[2].plot(x, motif * (0.6 + 0.4 * np.sin(6 * np.pi * x)), color=OUTC)
    ax[2].set(xlabel=r"$x$", ylabel="output",
              title=r"(c)  motif, modulated")
    fig.tight_layout(w_pad=1.8)
    save(fig, "hierarchy")


if __name__ == "__main__":
    fig_compose()
    fig_fold()
    fig_compose2d()
    fig_buildup()
    fig_regions()
    fig_representation()
    fig_hierarchy()
    print("figures ->", os.path.abspath(OUT))
