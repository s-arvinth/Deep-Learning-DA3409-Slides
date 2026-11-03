"""Shared style and helpers for the Class 11 figure scripts (Class11_Activations_Errors_Backprop_AD).

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
# Datasets for the Class 11 figures.


def inverse_problem(n=600, seed=1, noise=0.05):
    """A forward map that is one-to-one and its inverse, which is not.

    x = y + 0.3 sin(2 pi y) + noise, sampled with y uniform.  Reading the
    pair the other way round gives a target with up to three valid
    answers for one input -- the case a single-valued network cannot fit.
    """
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 1, n)
    x = y + 0.3 * np.sin(2 * np.pi * y) + noise * rng.normal(size=n)
    return x, y


def heteroscedastic(n=250, seed=3):
    """Noise whose width depends on the input."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    sd = 0.05 + 0.45 * (x + 1) / 2
    return x, np.sin(2.2 * x) + sd * rng.normal(size=n), sd


# =====================================================================
#  from src/common/models.py
# =====================================================================
# Activations, their derivatives, and a small network we can differentiate.
#
# Notation (the course convention, see _shared/NOTATION.md):
#     a^(l)_j            pre-activation of unit j in layer l
#     z^(l)_j = h(a)     its activation
#     delta^(l)_j        dE/da^(l)_j -- the 'error' backpropagated to that unit
#     W^(l), b^(l)       layer weights and biases


# ---- the activation zoo ---------------------------------------------
def relu(a):        return np.maximum(a, 0.0)
def drelu(a):       return (a > 0).astype(float)

def lrelu(a, s=0.1):  return np.where(a > 0, a, s * a)
def dlrelu(a, s=0.1): return np.where(a > 0, 1.0, s)

def elu(a, al=1.0):   return np.where(a > 0, a, al * (np.exp(np.minimum(a, 0)) - 1))
def delu(a, al=1.0):  return np.where(a > 0, 1.0, al * np.exp(np.minimum(a, 0)))

def sigmoid(a):     return 0.5 * (1.0 + np.tanh(0.5 * a))
def dsigmoid(a):    s = sigmoid(a); return s * (1 - s)

def dtanh(a):       return 1.0 - np.tanh(a) ** 2

def softplus(a):    return np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0)
def dsoftplus(a):   return sigmoid(a)

def _Phi(a):        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                              (a + 0.044715 * a ** 3)))
def gelu(a):        return a * _Phi(a)
def dgelu(a):
    e = 1e-5
    return (gelu(a + e) - gelu(a - e)) / (2 * e)

def silu(a):        return a * sigmoid(a)
def dsilu(a):       return sigmoid(a) * (1 + a * (1 - sigmoid(a)))


ACTIVATIONS = [
    ("ReLU",        r"$\max(0,a)$",            relu,     drelu),
    ("leaky ReLU",  r"$\max(0.1a,\,a)$",       lrelu,    dlrelu),
    ("ELU",         r"$a$ or $e^{a}-1$",       elu,      delu),
    ("softplus",    r"$\ln(1+e^{a})$",         softplus, dsoftplus),
    ("GELU",        r"$a\,\Phi(a)$",           gelu,     dgelu),
    ("SiLU / swish",r"$a\,\sigma(a)$",         silu,     dsilu),
    ("tanh",        r"$\tanh a$",              np.tanh,  dtanh),
    ("logistic",    r"$1/(1+e^{-a})$",         sigmoid,  dsigmoid),
]


# ---- a deep chain, for measuring how gradients travel ----------------
def layer_gradient_norms(L=20, M=64, gain=1.0, act="tanh", seed=0, N=256):
    """Forward a random input through L layers, backpropagate a unit
    signal, and record the gradient norm arriving at each layer."""
    rng = np.random.default_rng(seed)
    f, df = (np.tanh, dtanh) if act == "tanh" else (relu, drelu)
    Ws = [rng.normal(0, gain / np.sqrt(M), (M, M)) for _ in range(L)]
    Z = [rng.normal(size=(N, M))]
    A = []
    for W in Ws:
        A.append(Z[-1] @ W.T)
        Z.append(f(A[-1]))
    delta = rng.normal(size=(N, M)) / np.sqrt(M)      # unit signal at the top
    norms = []
    for l in range(L - 1, -1, -1):
        delta = delta * df(A[l])
        norms.append(np.sqrt((delta ** 2).sum(1)).mean())
        delta = delta @ Ws[l]
    return np.array(norms[::-1])


# ---- how many ReLU units are dead ------------------------------------
def dead_fraction(bias, M=512, D=64, N=2048, seed=0):
    """Fraction of units whose pre-activation is negative for EVERY input.

    Such a unit outputs zero on all data and has zero gradient, so no
    update can ever revive it.  A large negative bias -- which gradient
    descent can produce -- is what drives units into that state.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))
    W = rng.normal(0, 1.0 / np.sqrt(D), (M, D))
    A = X @ W.T + bias
    return float((A.max(0) <= 0).mean())
