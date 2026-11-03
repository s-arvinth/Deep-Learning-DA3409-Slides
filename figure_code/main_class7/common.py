"""Shared style and helpers for the Class 7 figure scripts (Class07_Deep_Networks_Universal_Approximation).

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
# Datasets for the Class 7 figures.


def two_moons(n=200, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, np.pi, n)
    A = np.stack([np.cos(t), np.sin(t)], 1)
    B = np.stack([1 - np.cos(t), 0.4 - np.sin(t)], 1)
    X = np.vstack([A, B]) + noise * rng.normal(size=(2 * n, 2))
    return X, np.hstack([np.zeros(n), np.ones(n)])


def target_2d(X1, X2):
    """The continuous target used in the universal-approximation figure."""
    r2 = X1 ** 2 + X2 ** 2
    return np.exp(-0.7 * r2) * np.sin(2.2 * X1 + 1.1 * X2) + 0.25 * X1


# =====================================================================
#  from src/common/models.py
# =====================================================================
# Models for Class 7, in the course notation.
#
#     x                      input
#     z^(l)_j                unit j of hidden layer l
#     a^(l)_j                its pre-activation
#     w^(l)_{ji}, w^(l)_{j0} weights and bias of layer l
#     L                      number of hidden layers (depth)
#     M                      units per hidden layer (width)
#     h(.)                   activation, ReLU throughout
#
# The two running networks compose to make the deep example:
#
#     f1:  z_1 = h(x + 1),  z_2 = h(x + 0.5),  z_3 = h(x - 0.5)
#          a   = -1 + 4 z_1 - 6 z_2 + 6 z_3        (a zigzag, 3 branches)
#
#     f2:  the same shape of network applied to f1's output


def relu(a):
    return np.maximum(a, 0.0)


# ---- network 1 and network 2 (scalar in, scalar out) -----------------
W1_1 = np.array([[1.0, 1.0], [0.5, 1.0], [-0.5, 1.0]])   # [bias, slope]
W2_1 = np.array([-1.0, 4.0, -6.0, 6.0])                  # [bias, weights]
W1_2 = np.array([[1.0, 1.0], [0.0, 1.0], [-0.5, 1.0]])
W2_2 = np.array([-1.0, 2.0, -3.5, 3.0])


def layer(x, W1, W2):
    """Return (pre-activations a, activations z, output)."""
    a = W1[:, 0][:, None] + W1[:, 1][:, None] * x[None, :]
    z = relu(a)
    return a, z, W2[0] + (W2[1:][:, None] * z).sum(axis=0)


def f1(x):
    return layer(x, W1_1, W2_1)[2]


def f2(x):
    return layer(x, W1_2, W2_2)[2]


# ---- the two-input network used for the 2-D composition --------------
V1 = np.array([[ 1.4, -0.6,  0.15],      # [w_j1, w_j2, w_j0]
               [-0.9, -1.1,  0.35],
               [ 0.3,  1.5,  0.10]])
V2 = np.array([0.0, 1.6, -1.4, 1.2])


def net2d(X1, X2):
    a = (V1[:, 0][:, None, None] * X1 + V1[:, 1][:, None, None] * X2
         + V1[:, 2][:, None, None])
    return a, V2[0] + (V2[1:][:, None, None] * relu(a)).sum(0)


# ---- the sawtooth: the witness for the depth separation --------------
def tooth(x):
    """T(x) = 2 h(x) - 4 h(x - 1/2)  maps [0,1] onto [0,1].

    Exactly a two-unit ReLU network, and T is a single 'tent': it folds
    [0,1] in half.  Composing it L times gives 2^L linear pieces.
    """
    return 2.0 * relu(x) - 4.0 * relu(x - 0.5)


def tooth_iter(x, L):
    y = x.copy()
    for _ in range(L):
        y = tooth(y)
    return y


# ---- parameter and region counts -------------------------------------
def params_shallow(M):
    """1 input, M hidden units, 1 output: 2M weights + M+1 biases/weights."""
    return 3 * M + 1


def params_deep(M, L):
    return 2 * M + (L - 1) * M * (M + 1) + M + 1


def regions_shallow(M):
    return M + 1.0


def regions_deep(M, L):
    return (M + 1.0) ** L
