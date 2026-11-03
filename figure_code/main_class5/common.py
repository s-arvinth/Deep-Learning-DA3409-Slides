"""Shared style, data and models for the Class 5 figure scripts.

Merged from Class05_Shallow_NN_Regression/B_Recreated_Figures/src/common/
(style.py, data.py, models.py): the Fira Sans type, the shared palette and
rcParams, save() with tick thinning, the datasets and the running models,
all unchanged apart from the output location.

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
from matplotlib.ticker import MaxNLocator, FixedLocator

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.abspath(os.path.join(HERE, "..", "..", "fonts"))
DEFAULT_OUT = os.path.join(HERE, "output")

# ---- SciencePlots preset, or the fallback the figures were built with ----
#      The deck's figs/ were rendered with the SciencePlots 'science' preset
#      active underneath RC below (inward ticks on all four sides, 0.05 in
#      save padding, mathtext tick labels), so the fallback restates those
#      preset values as well; RC then overrides sizes, minor ticks, legends.
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

# ---- Fira Sans, from the repository's root fonts/ --------------------------
for _f in sorted(glob.glob(os.path.join(FONTS, "*.ttf"))):
    try:
        font_manager.fontManager.addfont(_f)
    except Exception:
        pass


def out_dir():
    """Output folder: output/ next to the scripts, or --out DIR."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="directory to write the figure to (default: output/)")
    args, _ = p.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    return args.out


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
from matplotlib.colors import ListedColormap
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
    # --- type sizes: deliberately large.  Figures are scaled down a lot
    #     when placed on a slide, so what looks oversized in the PDF is
    #     what reads correctly from the back of a lecture theatre.
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
    # --- the science preset turns minor ticks on; at these type sizes
    #     they crowd the axis and the labels start to collide, so off.
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.major.pad":  4.0,
    "ytick.major.pad":  4.0,
    # --- legends: compact, opaque, never a box drawn over the data
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
    # --- a little breathing room so curves do not touch the frame
    "axes.xmargin":     0.04,
    "axes.ymargin":     0.08,
    "figure.dpi":       150,
    "pdf.fonttype":     42,      # embed TrueType, keep text selectable
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
}


def use():
    """Apply the shared style. Call once at the top of a figure module."""
    plt.rcParams.update(RC)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[L1, L2, OUTC, ACC, GREY])


def thin_ticks(fig, nx=5, ny=5):
    """Cap the number of major ticks on every linear axis of `fig`.

    At the type sizes this deck uses, matplotlib's default tick density
    makes the labels collide.  Log axes and colourbars are left alone.
    """
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d":
            continue
        if getattr(ax, "_colorbar", None) is not None:
            continue
        locked = getattr(ax, "_idl_locked", set())
        try:
            # never override ticks the figure set on purpose
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


def save(fig, name, nx=5, ny=5):
    """Write `fig` to <out>/<name>.pdf as vector output (out = output/ or --out).

    Thins the ticks first, so no figure ships with colliding labels.
    """
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
#  data  (was common/data.py): The datasets used across the Class 5 figures.
# =====================================================================
def linear_data(N=18, seed=5, noise=0.16, w0=0.35, w1=1.15):
    """N noisy samples of a line: y_n = w0 + w1 x_n + eps."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, N))
    y = w0 + w1 * x + noise * rng.normal(size=N)
    return x, y


def gaussian_cloud(N=160, seed=2, sd=0.16, w0=0.40, w1=1.00):
    """A dense cloud with constant Gaussian noise about a linear mean."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, N)
    y = w0 + w1 * x + sd * rng.normal(size=N)
    return x, y, (w0, w1, sd)


def target_wave(x):
    """The continuous target used in the approximation figures."""
    return np.sin(3.2 * np.pi * x) * np.exp(-1.2 * x)


# =====================================================================
#  models  (was common/models.py)
#
#  The running models for Class 5, in the notation used on the slides.
#
#  Notation (Bishop & Bishop, 2024, Ch. 4 and 6):
#      x            input          D  = input dimension
#      y            target         M  = hidden units
#      yhat         prediction     K  = outputs
#      a_j          pre-activation of hidden unit j
#      z_j = h(a_j) hidden unit (activation)
#      w^(1)_{ji}   weight from input i into hidden unit j;  w^(1)_{j0} its bias
#      w^(2)_{kj}   weight from hidden unit j into output k; w^(2)_{k0} its bias
#      phi_j(x)     a FIXED basis function -- never a parameter
# =====================================================================
def relu(a):
    return np.maximum(a, 0.0)


# ---------------------------------------------------------------------
#  the running scalar-input network:  D = 1, M = 3, K = 1
#     a_j    = w1[j,0] + w1[j,1] * x
#     z_j    = ReLU(a_j)
#     yhat   = w2[0] + sum_j w2[j+1] * z_j
# ---------------------------------------------------------------------
W1 = np.array([[ 0.55, 1.0],          # [w^(1)_{10}, w^(1)_{11}]
               [-0.15, 1.0],
               [-0.70, 1.0]])
W2 = np.array([-0.35, 1.6, -3.2, 2.4])   # [w^(2)_0, w^(2)_1, w^(2)_2, w^(2)_3]


def shallow(x, W1=W1, W2=W2):
    """Return (pre-activations a, activations z, output yhat)."""
    a = W1[:, 0][:, None] + W1[:, 1][:, None] * x[None, :]
    z = relu(a)
    return a, z, W2[0] + (W2[1:][:, None] * z).sum(0)


def joints(W1=W1):
    """Where each ReLU switches on: a_j = 0  =>  x = -w_{j0}/w_{j1}."""
    return [-b / w for b, w in W1]


# ---------------------------------------------------------------------
#  the running two-input network:  D = 2, M = 3, K = 1
# ---------------------------------------------------------------------
V1 = np.array([[ 1.6, -0.7,  0.10],   # [w^(1)_{j1}, w^(1)_{j2}, w^(1)_{j0}]
               [-1.0, -1.2,  0.30],
               [ 0.4,  1.7,  0.05]])
V2 = np.array([0.0, 1.5, -1.3, 1.1])


def shallow2d(X1, X2, V1=V1, V2=V2):
    a = (V1[:, 0][:, None, None] * X1 + V1[:, 1][:, None, None] * X2
         + V1[:, 2][:, None, None])
    z = relu(a)
    return a, V2[0] + (V2[1:][:, None, None] * z).sum(0)


# ---------------------------------------------------------------------
#  a ReLU network written as a linear model in learned features, so it
#  can be fitted in closed form for illustration purposes
# ---------------------------------------------------------------------
def relu_design(x, cuts):
    """Design matrix [ReLU(x - c_1) ... ReLU(x - c_M), x, 1]."""
    return np.stack([relu(x - c) for c in cuts] +
                    [x, np.ones_like(x)], axis=1)


def fit_relu(x, y, M):
    """Least-squares fit of an M-unit ReLU network with evenly spaced joints."""
    cuts = np.linspace(x.min(), x.max(), M + 2)[1:-1]
    A = relu_design(x, cuts)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return cuts, w
