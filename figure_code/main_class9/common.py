"""Shared style and helpers for the Class 9 figure scripts (Class09_MLP_Universal_Approximation_Efficiency).

Style: the house palette, type sizes, Fira fonts from the repository's
fonts/ folder, and save() which writes output/<name>.pdf (or --out DIR).
Models/data: ported unchanged from B_Recreated_Figures/src/common/.
Needs: numpy, matplotlib  (optional: sci_style)
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, FixedLocator
from matplotlib.colors import ListedColormap

# ---- locations -------------------------------------------------------
HERE   = os.path.dirname(os.path.abspath(__file__))
FONTS  = os.path.abspath(os.path.join(HERE, "..", "..", "fonts"))
OUTPUT = os.path.join(HERE, "output")

# ---- SciencePlots 'science' preset, or the fallback the figures were
#      actually built with -------------------------------------------
try:
    import sci_style                   # noqa: F401  (applies the preset)
except ImportError:
    plt.rcParams.update({"axes.grid": False,
                         "figure.facecolor": "white",
                         "axes.facecolor": "white"})
    # The PDFs in figs/ were rendered with the SciencePlots 'science'
    # preset active: inward ticks on all four sides, 0.05 in save padding,
    # thin default lines, and tick labels typeset as mathtext.  These are
    # the values of that preset which RC below does not override; with
    # them the scripts reproduce the deck's figures, without them they
    # do not (see NOTES.md).
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.625),
        "xtick.direction": "in", "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.5, "xtick.top": True,
        "ytick.direction": "in", "ytick.minor.size": 1.5,
        "ytick.minor.width": 0.5, "ytick.right": True,
        "grid.linewidth": 0.5, "lines.linewidth": 1.0,
        "savefig.pad_inches": 0.05,
        "axes.formatter.use_mathtext": True,
        "font.serif": ["Times"], "mathtext.fontset": "dejavuserif",
    })

# ---- Fira Sans / Fira Mono from the repository's fonts/ folder -------
for _f in glob.glob(os.path.join(FONTS, "*.ttf")):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass

# ---- one palette, used by every deck ---------------------------------
L1   = "#341651"      # indigo    — series 1 / fixed features / layer 1
L2   = "#1C7293"      # turquoise — series 2 / activations
OUTC = "#800000"      # maroon    — output / emphasis
ACC  = "#B8860B"      # amber     — series 4
GREY = "#6E6E6E"      # neutral   — references, guides, truth curves
BRANCH = [L1, L2, OUTC, ACC]
SEQ    = "viridis"    # sequential colormap for surfaces and densities
CAT    = "plasma"     # categorical ramp for ordered families of curves

# muted tints for activation-pattern regions (deliberately not a rainbow)
REGION_CMAP = ListedColormap([
    "#EFECF3", "#E1EAEF", "#F3E7E7", "#F6F1E5",
    "#E9EDE9", "#EDE6F0", "#E4EEF1", "#F8F4EC",
])

RC = {
    "text.usetex":      False,
    "font.family":      "Fira Sans",
    "mathtext.fontset": "custom",
    "mathtext.rm":      "Fira Sans",
    "mathtext.it":      "Fira Sans:italic",
    "mathtext.bf":      "Fira Sans:bold",
    "mathtext.sf":      "Fira Sans",
    "mathtext.tt":      "Fira Mono",
    "mathtext.cal":     "Fira Sans:italic",
    "mathtext.default": "regular",
    # type sizes: deliberately large (figures are scaled down on a slide)
    "font.size":        16.0,
    "axes.titlesize":   17.5,
    "axes.labelsize":   17.0,
    "xtick.labelsize":  14.5,
    "ytick.labelsize":  14.5,
    "legend.fontsize":  14.0,
    "axes.titlepad":    9.0,
    "axes.linewidth":   0.9,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.pad":  4.0,
    "ytick.major.pad":  4.0,
    # legends: compact, opaque, never a box drawn over the data
    "legend.frameon":       True,
    "legend.framealpha":    0.92,
    "legend.facecolor":     "white",
    "legend.edgecolor":     "none",
    "legend.borderpad":     0.35,
    "legend.labelspacing":  0.30,
    "legend.handlelength":  1.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 0.9,
    "legend.borderaxespad": 0.4,
    # a little breathing room so curves do not touch the frame
    "axes.xmargin":     0.04,
    "axes.ymargin":     0.08,
    "figure.dpi":       150,
    "pdf.fonttype":     42,      # embed TrueType, keep text selectable
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
}


def use():
    """Apply the shared style. Every figure script calls this first."""
    plt.rcParams.update(RC)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[L1, L2, OUTC, ACC, GREY])


def thin_ticks(fig, nx=5, ny=5):
    """Cap the number of major ticks on every linear axis of `fig`.

    Log axes, 3-D axes, colourbars and axes whose ticks were set on
    purpose (FixedLocator, or locked by integer_ticks) are left alone.
    """
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d":
            continue
        if getattr(ax, "_colorbar", None) is not None:
            continue
        locked = getattr(ax, "_idl_locked", set())
        try:
            if ("x" not in locked
                    and ax.get_xscale() == "linear"
                    and not isinstance(ax.xaxis.get_major_locator(),
                                       FixedLocator)):
                ax.xaxis.set_major_locator(MaxNLocator(nbins=nx, prune=None))
            if ("y" not in locked
                    and ax.get_yscale() == "linear"
                    and not isinstance(ax.yaxis.get_major_locator(),
                                       FixedLocator)):
                ax.yaxis.set_major_locator(MaxNLocator(nbins=ny, prune=None))
        except Exception:
            pass


def out_dir():
    """Output folder: `--out DIR` on the command line, else ./output/."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out", default=OUTPUT)
    args, _ = p.parse_known_args(sys.argv[1:])
    d = os.path.abspath(args.out)
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, name, nx=5, ny=5):
    """Thin the ticks, then write `fig` to <out>/<name>.pdf (vector)."""
    thin_ticks(fig, nx, ny)
    path = os.path.join(out_dir(), name + ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=400,
                metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote", path)
    return path


def square(ax):
    """Force a square data box — the preferred aspect for these decks."""
    ax.set_box_aspect(1.0)


def integer_ticks(ax, axis="x", nbins=6):
    """Whole-number ticks, for axes that count things (units, layers).

    Marks the axis so the automatic thinning in save() leaves it alone.
    """
    loc = MaxNLocator(nbins=nbins, integer=True)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(loc)
    locked = getattr(ax, "_idl_locked", set())
    locked.add(axis)
    ax._idl_locked = locked


def tidy3d(ax, ticks=(0.0, 0.5, 1.0), labelsize=11):
    """Three ticks per axis on a 3-D panel, which is all that fits."""
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticks(list(ticks))
    ax.tick_params(labelsize=labelsize, pad=0.5)


# =====================================================================
#  from src/common/data.py
# =====================================================================
# Datasets and target functions for the Class 9 figures.


def targets_1d(x):
    """The four targets used to illustrate universal approximation."""
    return {
        r"$f(x)=x^{2}$":            x ** 2,
        r"$f(x)=\sin(2\pi x)$":     np.sin(2 * np.pi * x),
        r"$f(x)=|x|$":              np.abs(x),
        r"$f(x)=\mathbb{1}[x>0]$":  (x > 0).astype(float),
    }


def smooth_target(x):
    """A single smooth target for the rate experiments."""
    return np.sin(3.0 * np.pi * x) * np.exp(-1.1 * x) + 0.4 * x


# =====================================================================
#  from src/common/models.py
# =====================================================================
# Models and counting formulas for Class 9.
#
# Notation (the course convention, see _shared/NOTATION.md):
#     x in R^D          input            D  input dimension
#     y, yhat           target, prediction
#     z^(l) = h(a^(l))  hidden units     M  width  (units per hidden layer)
#     W^(l), b^(l)      layer weights    L  depth  (number of hidden layers)
from math import comb


def relu(a):
    return np.maximum(a, 0.0)


# ---------------------------------------------------------------------
#  the tent map and its iterates -- the depth-separation witness
# ---------------------------------------------------------------------
def tooth(x):
    """T(x) = 2h(x) - 4h(x - 1/2): a two-unit ReLU network folding [0,1]."""
    return 2.0 * relu(x) - 4.0 * relu(x - 0.5)


def tooth_iter(x, L):
    y = np.asarray(x, float).copy()
    for _ in range(L):
        y = tooth(y)
    return y


# ---------------------------------------------------------------------
#  parameter counts:  D inputs, L hidden layers of width M, 1 output
# ---------------------------------------------------------------------
def n_params(M, L, D=1):
    return M * (D + 1) + (L - 1) * M * (M + 1) + (M + 1)


# ---------------------------------------------------------------------
#  linear-region counts
# ---------------------------------------------------------------------
def regions_shallow(M, D=1):
    """Zaslavsky: M hyperplanes in general position in R^D."""
    return float(sum(comb(int(M), j) for j in range(0, D + 1)))


def regions_deep_lower(M, L, D=1):
    """Montufar et al. (2014) lower bound for width M >= D."""
    if M < D:
        return np.nan
    return float((M // D) ** (D * (L - 1))) * regions_shallow(M, D)


def regions_deep_upper(M, L, D=1):
    """A simple product upper bound (cf. Serra et al., 2018)."""
    return float(regions_shallow(M, D) ** L)


# ---------------------------------------------------------------------
#  approximation rates
# ---------------------------------------------------------------------
def barron_rate(M, C=1.0):
    """Barron (1993): ||f - f_M||_2 <= C / sqrt(M), independent of D."""
    return C / np.sqrt(M)


def sobolev_rate(W, D, s=2.0, C=1.0):
    """DeVore, Howard & Micchelli (1989): W parameters cannot beat
    W^{-s/D} uniformly over the Sobolev ball of smoothness s."""
    return C * np.asarray(W, float) ** (-s / D)


# ---------------------------------------------------------------------
#  VC dimension of a ReLU network (Bartlett et al., 2019): Theta(W L log W)
# ---------------------------------------------------------------------
def vc_dim(W, L):
    W = np.asarray(W, float)
    return W * L * np.log2(np.maximum(W, 2.0))


# ---------------------------------------------------------------------
#  best continuous piecewise-linear approximation with P pieces
# ---------------------------------------------------------------------
def best_pwl(x, y, P):
    """Least-squares fit of a continuous piecewise-linear function with
    P pieces on an even grid -- the best a width-(P-1) shallow ReLU
    network can do with joints in those places."""
    knots = np.linspace(x.min(), x.max(), P + 1)[1:-1]
    A = np.stack([relu(x - c) for c in knots] + [x, np.ones_like(x)], 1)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ w
