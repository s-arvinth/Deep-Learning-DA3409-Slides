"""Shared style, data and models for the Class 6 figure scripts.

Merged from Class06_Shallow_NN_Classification/B_Recreated_Figures/src/common/
(style.py, data.py, models.py): the Fira Sans type, the shared palette and
rcParams, save() with tick thinning, the datasets and the running models,
all unchanged apart from the output location.

Needs: numpy, matplotlib  (optional: sci_style)
"""
import argparse
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
#      Only the faces the deck's own fonts/ folder carried when figs/ was
#      rendered.  It had no FiraSans-Bold, so $\mathbf{...}$ came out in
#      Fira Sans Regular; registering Bold as well would change those glyphs.
FACES = ["FiraSans-Regular.ttf", "FiraSans-Italic.ttf", "FiraSans-Light.ttf",
         "FiraSans-LightItalic.ttf", "FiraMono-Regular.ttf",
         "FiraMono-Bold.ttf"]
for _f in [os.path.join(FONTS, f) for f in FACES]:
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
#  data  (was common/data.py): Datasets used across the Class 6 (classification) figures.
# =====================================================================
def two_gaussians(N=120, seed=3, sep=2.4, cov=((1.0, 0.55), (0.55, 0.9))):
    """Two Gaussian classes with a SHARED covariance -- the setting in
    which the posterior is exactly a logistic sigmoid."""
    rng = np.random.default_rng(seed)
    S = np.asarray(cov)
    m1 = np.array([-sep / 2, -0.35])
    m2 = np.array([+sep / 2, +0.35])
    L = np.linalg.cholesky(S)
    X1 = m1 + rng.normal(size=(N, 2)) @ L.T
    X2 = m2 + rng.normal(size=(N, 2)) @ L.T
    X = np.vstack([X1, X2])
    y = np.r_[np.zeros(N), np.ones(N)]
    return X, y, (m1, m2, S)


def three_gaussians(N=90, seed=8):
    rng = np.random.default_rng(seed)
    means = np.array([[-1.7, -1.1], [1.8, -0.9], [0.1, 1.9]])
    S = np.array([[0.85, 0.25], [0.25, 0.70]])
    L = np.linalg.cholesky(S)
    X = np.vstack([m + rng.normal(size=(N, 2)) @ L.T for m in means])
    y = np.repeat(np.arange(3), N)
    return X, y, (means, S)


def separable_1d(N=20, seed=1, outliers=False):
    """One-dimensional two-class data; optionally with a cluster of
    distant but correctly-labelled points."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(-1.2, 0.55, N)
    x1 = rng.normal(1.2, 0.55, N)
    if outliers:
        x1 = np.r_[x1, rng.normal(6.5, 0.35, N // 2)]
    x = np.r_[x0, x1]
    y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    return x, y


def noisy_xor(N=70, seed=11, spread=0.42):
    """Four clusters in XOR arrangement: not linearly separable, but a
    two-unit hidden layer is enough."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]], float)
    labels = np.array([0, 0, 1, 1])
    X = np.vstack([c + spread * rng.normal(size=(N, 2)) for c in centres])
    y = np.repeat(labels, N).astype(float)
    return X, y


def regression_cloud(N=140, seed=4, sd=0.22):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, N)
    mean = 0.35 + 1.25 * x
    return x, mean + sd * rng.normal(size=N), sd


# =====================================================================
#  models  (was common/models.py)
#
#  Models for Class 6, written in the course notation.
#
#      x            input          D  = input dimension
#      y            target         M  = hidden units
#      yhat         prediction     K  = classes / outputs
#      a            pre-activation of an OUTPUT unit
#      sigma(a)     logistic sigmoid       -> Bernoulli parameter
#      softmax(a)   normalised exponential -> categorical parameters
#      w^(1), w^(2) all learnable weights
# =====================================================================
def sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))        # overflow-safe


def dsigmoid(a):
    s = sigmoid(a)
    return s * (1.0 - s)


def softmax(A, axis=-1):
    A = A - A.max(axis=axis, keepdims=True)      # shift invariance
    E = np.exp(A)
    return E / E.sum(axis=axis, keepdims=True)


def relu(a):
    return np.maximum(a, 0.0)


# ---------------------------------------------------------------------
#  logistic regression, fitted by gradient descent on cross-entropy
# ---------------------------------------------------------------------
def fit_logistic(X, y, steps=4000, lr=0.25):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        yhat = sigmoid(Xb @ w)
        w -= lr * (Xb.T @ (yhat - y)) / len(X)   # grad = sum (yhat - y) x
    return w


def fit_least_squares(X, y):
    """Least squares on the SAME 0/1 targets -- the thing we argue against."""
    Xb = np.c_[np.ones(len(X)), X]
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return w


# ---------------------------------------------------------------------
#  the exact Gaussian-class-conditional posterior (Bishop, 2024, S5.3)
#     shared covariance  =>  p(C1|x) = sigma(w'x + w0)
# ---------------------------------------------------------------------
def gaussian_posterior_weights(m1, m2, S, p1=0.5, p2=0.5):
    Si = np.linalg.inv(S)
    w = Si @ (m2 - m1)
    w0 = (-0.5 * m2 @ Si @ m2 + 0.5 * m1 @ Si @ m1 + np.log(p2 / p1))
    return w, w0


# ---------------------------------------------------------------------
#  a shallow network with M hidden ReLU units and a sigmoid output,
#  trained by full-batch gradient descent on cross-entropy
# ---------------------------------------------------------------------
def fit_shallow_classifier(X, y, M=2, steps=6000, lr=0.35, seed=0):
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / D), (M, D))
    b1 = np.zeros(M)
    W2 = rng.normal(0, np.sqrt(2.0 / M), M)
    b2 = 0.0
    N = len(X)
    for _ in range(steps):
        A1 = X @ W1.T + b1
        Z = relu(A1)
        a = Z @ W2 + b2
        yhat = sigmoid(a)
        d = (yhat - y) / N                        # dE/da  =  yhat - y
        gW2 = Z.T @ d
        gb2 = d.sum()
        dZ = np.outer(d, W2) * (A1 > 0)
        gW1 = dZ.T @ X
        gb1 = dZ.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2


def shallow_forward(X, W1, b1, W2, b2):
    Z = relu(X @ W1.T + b1)
    return Z, sigmoid(Z @ W2 + b2)


# ---------------------------------------------------------------------
#  fixed Gaussian basis functions (Bishop, 2024, Fig. 5.15)
# ---------------------------------------------------------------------
def gaussian_basis(X, centres, s=0.65):
    d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * s ** 2))
