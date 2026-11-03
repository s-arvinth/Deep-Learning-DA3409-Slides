"""Shared style and helpers for the Class 2 figure scripts.

Ported from Class02_Intro_Neural_Networks/B_Recreated_Figures/src/make_figures.py:
Fira Sans type, the SciencePlots preset via sci_style (with the fallback the
figures were actually built with), the deck palette and a save() helper.

Needs: matplotlib  (optional: sci_style)
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.abspath(os.path.join(HERE, "..", "..", "fonts"))
DEFAULT_OUT = os.path.join(HERE, "output")

# ---- SciencePlots preset, or the fallback the figures were built with ----
#      The deck's figs/ were rendered with the SciencePlots 'science' preset
#      active (inward ticks on all four sides, minor ticks, 0.05 in save
#      padding, mathtext tick labels), so the fallback restates those preset
#      values as well; with them the outputs match figs/ pixel for pixel.
SCIENCE_RC = {
    "axes.formatter.use_mathtext": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.5, "grid.linewidth": 0.5, "lines.linewidth": 1.0,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "figure.figsize": (3.5, 2.625),
}
try:
    import sci_style  # noqa: F401  (SciencePlots 'science' preset)
except ImportError:
    plt.rcParams.update(SCIENCE_RC)
    plt.rcParams.update({"axes.grid": False, "figure.facecolor": "white",
                         "axes.facecolor": "white"})

# ---- Fira Sans everywhere, matching the slides ----------------------------
for _f in sorted(glob.glob(os.path.join(FONTS, "*.ttf"))):
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

# ---- shared palette --------------------------------------------------------
L1   = "#341651"      # indigo    - inputs / class 0 / first series
L2   = "#1C7293"      # turquoise - activation / second series
OUTC = "#800000"      # maroon    - output / class 1 / emphasis
ACC  = "#B8860B"      # amber     - fourth series
GREY = "#6E6E6E"
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[L1, L2, OUTC, ACC, GREY])


def out_dir():
    """Output folder: output/ next to the scripts, or --out DIR."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="directory to write the figure to (default: output/)")
    args, _ = p.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    return args.out


def save(fig, name, ext="pdf"):
    """Write fig to <out>/<name>.<ext> (vector; shaded surfaces at 400 dpi)."""
    path = os.path.join(out_dir(), f"{name}.{ext}")
    kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {}
    fig.savefig(path, bbox_inches="tight", dpi=400, **kw)
    plt.close(fig)
    print("wrote", path)
    return path


# ---- the 3-input cube cut by threshold planes (cube_or, cube_majority) -----
def _cube(ax):
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
    return save(fig, name)
