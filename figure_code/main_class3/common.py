"""Shared style and helpers for the Class 3 figure scripts.

Ported from Class03_Perceptrons_PLA/B_Recreated_Figures/src/make_figures.py:
Fira Sans type, the SciencePlots preset via sci_style (with the fallback the
figures were actually built with), the deck palette and a save() helper.

Needs: numpy, matplotlib  (optional: sci_style)
"""
import argparse
import glob
import os

import numpy as np
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
L1   = "#341651"      # indigo    - class N / first series
L2   = "#1C7293"      # turquoise - boundary / second series
OUTC = "#800000"      # maroon    - class P / emphasis
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



# ---- plotting helpers ------------------------------------------------------
def scatter_PN(ax, P, N, s=42):
    ax.scatter(*P.T, s=s + 8, marker="^", facecolor=OUTC, edgecolor="white",
               lw=0.7, zorder=4, label=r"$P$ (target 1)")
    ax.scatter(*N.T, s=s, marker="o", facecolor="white", edgecolor=L1,
               lw=1.4, zorder=4, label=r"$N$ (target 0)")


def boundary(ax, w, xs, **kw):
    """draw w0 + w1 x1 + w2 x2 = 0"""
    if abs(w[2]) > 1e-9:
        ax.plot(xs, -(w[0] + w[1] * xs) / w[2], **kw)


# ---- the perceptron learning algorithm -------------------------------------
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


# ---- the AND error surface (Rojas, Fig. 4.5: fixed threshold 1) ------------
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
