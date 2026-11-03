"""Shared style and helpers for the Class 13 figure scripts (Class13_Gradient_Descent_Optimization).

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
#  from src/common/models.py
# =====================================================================
# Models and optimisers used by the Class 13 figures.
#
# Everything here is deliberately small and explicit so a student can read
# the update rule off the code and match it to the slide.
#
# Notation follows the frozen course convention (see _shared/NOTATION.md):
#
#     w       the parameter vector, whatever the book calls it
#     E(w)    the error function
#     eta     the learning rate
#     B       the mini-batch size
#
# Prince writes the parameters as `phi`, the learning rate as `alpha` and
# the loss as `L[.]`.  Those are translated here once, at the source, so no
# figure and no slide has to carry two notations.


# =====================================================================
#  A two-parameter quadratic error surface
# =====================================================================
def quadratic_E(W1, W2, lam=(1.0, 8.0), centre=(0.0, 0.0), rot=0.0):
    """E(w) = E* + 1/2 sum_i lambda_i alpha_i^2  (Bishop Eq. 7.11).

    `lam` are the Hessian eigenvalues, `rot` rotates the eigenvectors
    away from the coordinate axes so the picture is not degenerate.
    """
    c, s = np.cos(rot), np.sin(rot)
    d1, d2 = W1 - centre[0], W2 - centre[1]
    a1 = c * d1 + s * d2
    a2 = -s * d1 + c * d2
    return 0.5 * (lam[0] * a1 ** 2 + lam[1] * a2 ** 2)


def quadratic_grad(w, lam=(1.0, 8.0), centre=(0.0, 0.0), rot=0.0):
    """Gradient of `quadratic_E`, i.e. H (w - w*)."""
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, s], [-s, c]])
    H = R.T @ np.diag(lam) @ R
    return H @ (np.asarray(w, float) - np.asarray(centre, float))


def hessian(lam=(1.0, 8.0), rot=0.0):
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, s], [-s, c]])
    return R.T @ np.diag(lam) @ R


# =====================================================================
#  A non-convex two-parameter surface, for the landscape figures
# =====================================================================
def bumpy_E(W1, W2):
    """A smooth non-convex surface with two minima and a saddle.

    Used only to make the geometry of Bishop Fig. 7.1 concrete: one
    local minimum, one global minimum, and a gradient arrow in between.
    """
    return (0.6 * (W1 ** 2 + W2 ** 2)
            - 1.9 * np.exp(-((W1 + 1.15) ** 2 + (W2 + 0.75) ** 2) / 0.55)
            - 2.9 * np.exp(-((W1 - 1.20) ** 2 + (W2 - 0.65) ** 2) / 0.55))


def bumpy_grad(w):
    w1, w2 = float(w[0]), float(w[1])
    g1 = 1.2 * w1
    g2 = 1.2 * w2
    for amp, c1, c2 in ((-1.9, -1.15, -0.75), (-2.9, 1.20, 0.65)):
        e = np.exp(-((w1 - c1) ** 2 + (w2 - c2) ** 2) / 0.55)
        g1 += amp * e * (-2 * (w1 - c1) / 0.55)
        g2 += amp * e * (-2 * (w2 - c2) / 0.55)
    return np.array([g1, g2])


# =====================================================================
#  Prince's Gabor model  (Prince 2023, Eq. 6.8)
# =====================================================================
def gabor(x, w):
    """f(x, w) = sin(w0 + 0.06 w1 x) exp(-(w0 + 0.06 w1 x)^2 / 32)."""
    u = w[0] + 0.06 * w[1] * np.asarray(x, float)
    return np.sin(u) * np.exp(-(u ** 2) / 32.0)


def gabor_data(seed=7, n=28, w_true=(0.0, 16.6), noise=0.06):
    """Prince's training set: 28 points, uniform x, Gabor plus noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-15.0, 15.0, n)
    y = gabor(x, np.array(w_true)) + noise * rng.standard_normal(n)
    return x, y


def gabor_loss(w0, w1, x, y):
    """Least-squares loss (Prince Eq. 6.9), vectorised over a w-grid."""
    w0 = np.atleast_1d(w0)
    u = w0[..., None] + 0.06 * np.atleast_1d(w1)[..., None] * x
    pred = np.sin(u) * np.exp(-(u ** 2) / 32.0)
    return ((pred - y) ** 2).sum(axis=-1)


def gabor_grad(w, x, y):
    """Analytic gradient of the Gabor least-squares loss."""
    u = w[0] + 0.06 * w[1] * x
    e = np.exp(-(u ** 2) / 32.0)
    f = np.sin(u) * e
    dfdu = np.cos(u) * e + np.sin(u) * e * (-u / 16.0)
    r = 2.0 * (f - y)
    return np.array([(r * dfdu).sum(), (r * dfdu * 0.06 * x).sum()])


# =====================================================================
#  Optimisers
# =====================================================================
def gradient_descent(grad, w0, eta, steps):
    """Batch gradient descent, Bishop Eq. 7.16:  w <- w - eta grad E."""
    w = np.asarray(w0, float).copy()
    path = [w.copy()]
    for _ in range(steps):
        w = w - eta * grad(w)
        path.append(w.copy())
    return np.array(path)


def sgd(grad_batch, w0, eta, steps, n, batch, seed=0):
    """Mini-batch SGD, Bishop Alg. 7.2 / Prince Eq. 6.10.

    Draws without replacement and reshuffles at the end of each epoch,
    exactly as Algorithm 7.2 specifies.
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, float).copy()
    path = [w.copy()]
    order = rng.permutation(n)
    k = 0
    for _ in range(steps):
        if k + batch > n:
            order = rng.permutation(n)
            k = 0
        idx = order[k:k + batch]
        k += batch
        w = w - eta * grad_batch(w, idx) * (n / len(idx))
        path.append(w.copy())
    return np.array(path)


# =====================================================================
#  The mini-batch gradient as an estimator  (Bishop Section 7.2.4)
# =====================================================================
def batch_gradient_samples(per_point_grads, batch, draws, seed=0):
    """Sample the mini-batch gradient estimate `draws` times.

    `per_point_grads` is the (N, d) array of per-example gradients.  The
    mini-batch estimate is their mean over a random subset of size
    `batch`; its standard error is sigma / sqrt(batch), which is the
    quantity Bishop Section 7.2.4 uses to argue for diminishing returns.
    """
    rng = np.random.default_rng(seed)
    n = len(per_point_grads)
    out = np.empty((draws, per_point_grads.shape[1]))
    for i in range(draws):
        idx = rng.choice(n, size=batch, replace=False)
        out[i] = per_point_grads[idx].mean(axis=0)
    return out
