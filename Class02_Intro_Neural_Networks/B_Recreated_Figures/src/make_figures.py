#!/usr/bin/env python3
# =====================================================================
#  Figures for Class 2 — Introduction to Neural Networks
#
#  Style: SciencePlots "science" preset via sci_style, Fira Sans type.
#  Run:   python3 make_figures.py          -> writes ../figs/*.pdf
# =====================================================================
import sci_style                      # SciencePlots 'science' preset
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
L1   = "#341651"      # indigo   — inputs / class 0 / first series
L2   = "#1C7293"      # turquoise— activation / second series
OUTC = "#800000"      # maroon   — output / class 1 / emphasis
ACC  = "#B8860B"      # amber    — fourth series
GREY = "#6E6E6E"
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[L1, L2, OUTC, ACC, GREY])

def save(fig, name):
    p = os.path.join(OUT, name + ".pdf")
    fig.savefig(p, bbox_inches="tight", dpi=400)  # vector; shaded
                                                 # surfaces at 400dpi
    plt.close(fig)
    print("wrote", os.path.basename(p))


# =====================================================================
#  1 — the action potential
# =====================================================================
def fig_action_potential():
    t = np.linspace(0, 10, 2000)
    v = np.full_like(t, -70.0)                       # resting, mV
    # sub-threshold bumps that fail to fire
    for c in (1.2, 2.4):
        v += 6 * np.exp(-((t - c) ** 2) / 0.05)
    # the spike itself
    spike = 100 * np.exp(-((t - 5.0) ** 2) / 0.010)
    after = -18 * np.exp(-((t - 5.35) ** 2) / 0.050)  # refractory dip
    v = v + spike + after

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axhline(-55, color=GREY, ls="--", lw=1.0)
    ax.text(9.6, -53, "threshold", color=GREY, ha="right", fontsize=9)
    ax.axhline(-70, color=GREY, ls=":", lw=0.8)
    ax.text(0.1, -68.5, "resting", color=GREY, fontsize=9)
    ax.plot(t, v, color=OUTC, lw=1.6)
    ax.annotate("sub-threshold:\nno spike", xy=(2.4, -62), xytext=(2.9, -30),
                fontsize=9, color=L1,
                arrowprops=dict(arrowstyle="->", color=L1, lw=0.8))
    ax.annotate("action potential", xy=(5.0, 22), xytext=(6.1, 8),
                fontsize=9, color=OUTC,
                arrowprops=dict(arrowstyle="->", color=OUTC, lw=0.8))
    ax.annotate("refractory dip", xy=(5.45, -84), xytext=(6.6, -95),
                fontsize=9, color=L2,
                arrowprops=dict(arrowstyle="->", color=L2, lw=0.8))
    ax.set(xlabel="time (ms)", ylabel="membrane potential (mV)",
           ylim=(-105, 45), xlim=(0, 10))
    save(fig, "action_potential")


# =====================================================================
#  2 — the step activation, and why it is the essential feature
# =====================================================================
def fig_step():
    z = np.linspace(-3, 3, 2000)
    theta = 0.8
    step = (z >= theta).astype(float)

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 2.9))

    ax[0].plot(z, step, color=OUTC, lw=1.8)
    ax[0].axvline(theta, color=GREY, ls=":", lw=0.8)
    ax[0].text(theta + 0.08, 0.45, r"$\theta$", color=GREY, fontsize=11)
    ax[0].set(xlabel=r"total excitation $z$", ylabel="output",
              ylim=(-0.15, 1.2),
              title=r"(a)  step: fires iff $z \geq \theta$")

    for k, c in zip([1, 4, 16], [L1, L2, OUTC]):
        ax[1].plot(z, 1 / (1 + np.exp(-k * (z - theta))), color=c,
                   lw=1.5, label=rf"$k={k}$")
    ax[1].plot(z, step, color=GREY, lw=1.0, ls="--", label="step")
    ax[1].axvline(theta, color=GREY, ls=":", lw=0.8)
    ax[1].legend(loc="upper left")
    ax[1].set(xlabel=r"$z$", ylabel="output", ylim=(-0.15, 1.2),
              title=r"(b)  $\sigma(k(z-\theta))$ sharpens to the step")
    fig.tight_layout(w_pad=2.0)
    save(fig, "step_activation")


# =====================================================================
#  3 — AND, OR, XOR in the 2-D input square
# =====================================================================
def fig_logic_2d():
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tables = {
        "AND": np.array([0, 0, 0, 1]),
        "OR":  np.array([0, 1, 1, 1]),
        "XOR": np.array([0, 1, 1, 0]),
    }
    # thresholds that realise AND and OR with unit weights
    lines = {"AND": 1.5, "OR": 0.5, "XOR": None}

    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.1))
    for a, (name, y) in zip(ax, tables.items()):
        a.scatter(*pts[y == 0].T, s=90, marker="o", facecolor="white",
                  edgecolor=L1, lw=1.6, zorder=3, label="output 0")
        a.scatter(*pts[y == 1].T, s=110, marker="^", facecolor=OUTC,
                  edgecolor="white", lw=1.0, zorder=3, label="output 1")
        th = lines[name]
        if th is not None:
            xs = np.linspace(-0.45, 1.45, 10)
            a.plot(xs, th - xs, color=L2, lw=1.6,
                   label=rf"$x_1+x_2=\theta$, $\theta={th}$")
        else:
            xs = np.linspace(-0.45, 1.45, 10)
            for th_bad in (0.5, 1.5):
                a.plot(xs, th_bad - xs, color=GREY, lw=0.9, ls="--",
                       alpha=.7)
            a.text(0.5, -0.33, "no single line separates",
                   color=GREY, fontsize=8.5, ha="center")
        a.set(xlim=(-0.45, 1.45), ylim=(-0.45, 1.45),
              xlabel=r"$x_1$", ylabel=r"$x_2$", title=f"({'abc'[list(tables).index(name)]})  {name}")
        a.set_xticks([0, 1]); a.set_yticks([0, 1])
        a.set_aspect("equal", "box")
    ax[0].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(w_pad=1.6)
    save(fig, "logic_2d")


# =====================================================================
#  4 — the 3-input cube: a threshold unit cuts it with a plane
# =====================================================================
def _cube(ax):
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if np.abs(v[i] - v[j]).sum() == 1:
                ax.plot(*zip(v[i], v[j]), color="#B0B0B0", lw=1.0,
                        zorder=1)
    return v

def _plane(ax, theta, color, alpha=0.45):
    """Solid-ish plane x1+x2+x3 = theta, clipped to the unit cube,
    with its outline drawn so the cut is unmistakable."""
    g = np.linspace(0, 1, 90)
    X, Y = np.meshgrid(g, g)
    Z = theta - X - Y
    Z[(Z < 0) | (Z > 1)] = np.nan
    surf = ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                           shade=False, linewidth=0, antialiased=True,
                           zorder=2)
    surf.set_rasterized(True)
    # outline of the cut polygon = where the plane meets the cube faces
    pts = []
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        for coord in range(3):
            p = [None, None, None]
            idx = [i for i in range(3) if i != coord]
            p[idx[0]], p[idx[1]] = a, b
            t = theta - a - b
            if 0 <= t <= 1:
                p[coord] = t
                pts.append(tuple(p))
    pts = sorted(set(pts),
                 key=lambda p: np.arctan2(p[1] - theta/3, p[0] - theta/3))
    if len(pts) >= 3:
        loop = list(pts) + [pts[0]]
        ax.plot(*zip(*loop), color=color, lw=1.8, zorder=6)

def fig_cube(name, planes, fn, note):
    """planes: list of (theta, colour, label)"""
    fig = plt.figure(figsize=(5.2, 3.8))
    ax = fig.add_subplot(111, projection="3d")
    v = _cube(ax)
    y = np.array([fn(p) for p in v])
    for th, c, _lab in planes:
        _plane(ax, th, c)
    # vertices: filled triangles fire, hollow circles do not
    ax.scatter(*v[y == 1].T, s=95, marker="^", color=OUTC,
               edgecolor="white", lw=1.0, depthshade=False, zorder=8)
    ax.scatter(*v[y == 0].T, s=80, marker="o", facecolor="white",
               edgecolor=L1, lw=1.8, depthshade=False, zorder=8)
    # a proxy legend carrying the plane equations
    handles = [plt.Line2D([], [], marker="^", ls="", color=OUTC,
                          mec="white", ms=9, label="fires (output 1)"),
               plt.Line2D([], [], marker="o", ls="", mfc="white",
                          mec=L1, mew=1.6, ms=8, label="silent (output 0)")]
    for th, c, lab in planes:
        handles.append(plt.Line2D([], [], color=c, lw=2.6, label=lab))
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(-0.14, 1.02), fontsize=8.5,
              handlelength=1.6, borderpad=0.3, labelspacing=0.35)
    ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", zlabel=r"$x_3$",
           xticks=[0, 1], yticks=[0, 1], zticks=[0, 1],
           xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    ax.tick_params(labelsize=8.5, pad=-1)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.label.set_size(10)
    ax.text2D(0.52, -0.02, note, transform=ax.transAxes, fontsize=9,
              color=GREY, ha="center")
    ax.view_init(elev=18, azim=36)
    ax.set_box_aspect((1, 1, 1), zoom=0.88)
    fig.subplots_adjust(left=0.04, right=0.99, top=1.02, bottom=0.06)
    save(fig, name)


# =====================================================================
#  5 — how rare are the linearly separable functions?
# =====================================================================
def fig_separable():
    n    = np.array([1, 2, 3, 4, 5])
    tot  = np.array([4, 16, 256, 65536, 4294967296], dtype=float)
    sep  = np.array([4, 14, 104, 1882, 94572], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.1))
    w = 0.36
    ax[0].bar(n - w/2, tot, w, color=L1, label="all Boolean functions")
    ax[0].bar(n + w/2, sep, w, color=OUTC, label="linearly separable")
    ax[0].set(yscale="log", xlabel=r"number of inputs $n$",
              ylabel="count", xticks=n,
              title="(a)  how many functions")
    ax[0].legend(loc="upper left")

    frac = 100 * sep / tot
    ax[1].plot(n, frac, color=OUTC, marker="o", ms=6, mew=0.8, mec="white")
    for xi, yi in zip(n, frac):
        dx, ha = (-10, "right") if xi == n[-1] else (0, "center")
        ax[1].annotate(f"{yi:.3g}%", (xi, yi), textcoords="offset points",
                       xytext=(dx, 9), ha=ha, fontsize=8.5, color=GREY)
    ax[1].set_xlim(0.6, 5.4)
    ax[1].set(yscale="log", xlabel=r"number of inputs $n$",
              ylabel="separable (\\% of all)", xticks=n,
              title="(b)  the fraction collapses")
    fig.tight_layout(w_pad=2.0)
    save(fig, "separable_fraction")


# =====================================================================
#  6 — a network of simple units approximating a function
# =====================================================================
def fig_approx():
    x = np.linspace(-1, 1, 1500)
    target = np.sin(2.4 * np.pi * x) * np.exp(-x**2)

    def fit_steps(m):
        """least-squares fit of m shifted step (threshold) units"""
        cuts = np.linspace(-1, 1, m + 2)[1:-1]
        A = np.stack([(x >= c).astype(float) for c in cuts] +
                     [np.ones_like(x)], 1)
        w, *_ = np.linalg.lstsq(A, target, rcond=None)
        return A @ w

    fig, ax = plt.subplots(1, 3, figsize=(9.8, 2.8))
    for a, m, c in zip(ax, [3, 10, 40], [L1, L2, OUTC]):
        a.plot(x, target, color=GREY, lw=1.6, ls="--", label="target")
        a.plot(x, fit_steps(m), color=c, lw=1.5, label=f"{m} units")
        a.legend(loc="upper right", fontsize=8.5)
        a.set(xlabel=r"$x$", ylabel=r"$f(x)$",
              title=rf"({'abc'[[3,10,40].index(m)]})  {m} threshold units")
    fig.tight_layout(w_pad=1.7)
    save(fig, "approximation")


if __name__ == "__main__":
    fig_action_potential()
    fig_step()
    fig_logic_2d()
    fig_cube("cube_or",
             [(0.5, L2, r"$x_1{+}x_2{+}x_3 = \frac{1}{2}$  (OR, $\theta{=}1$)")],
             lambda p: int(p.sum() >= 1),
             "one plane separates the 7 firing corners from the 1 silent corner")
    fig_cube("cube_majority",
             [(0.5, L2, r"$x_1{+}x_2{+}x_3=\frac{1}{2}$  (OR)"),
              (1.5, ACC, r"$x_1{+}x_2{+}x_3=\frac{3}{2}$  (majority)")],
             lambda p: int(p.sum() >= 2),
             "the two planes are parallel: only the offset changes")
    fig_separable()
    fig_approx()
    print("figures ->", os.path.abspath(OUT))
